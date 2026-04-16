library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity pcie_dma is
    Port (
        -- Clock and reset
        clk : in STD_LOGIC;
        reset : in STD_LOGIC;

        -- PCIe interface (Gen4 x16, 256-bit data path)
        pcie_clk : in STD_LOGIC;
        pcie_rst_n : in STD_LOGIC;
        pcie_rx_data : in STD_LOGIC_VECTOR (255 downto 0);
        pcie_rx_valid : in STD_LOGIC;
        pcie_tx_data : out STD_LOGIC_VECTOR (255 downto 0);
        pcie_tx_valid : out STD_LOGIC;
        pcie_tx_ready : in STD_LOGIC;

        -- DMA control interface
        dma_start : in STD_LOGIC;
        dma_direction : in STD_LOGIC;  -- 0 = host->card, 1 = card->host
        dma_length : in UNSIGNED (31 downto 0);
        host_addr : in UNSIGNED (63 downto 0);
        card_addr : in UNSIGNED (31 downto 0);

        -- Scatter-gather list interface
        sg_addr : in UNSIGNED (63 downto 0);
        sg_length : in UNSIGNED (31 downto 0);
        sg_valid : in STD_LOGIC;

        -- Status and interrupts
        dma_done : out STD_LOGIC;
        dma_error : out STD_LOGIC;
        interrupt_req : out STD_LOGIC;

        -- Memory interface (to order book and matching engine)
        mem_addr : out UNSIGNED (31 downto 0);
        mem_data_in : in STD_LOGIC_VECTOR (255 downto 0);
        mem_data_out : out STD_LOGIC_VECTOR (255 downto 0);
        mem_write_en : out STD_LOGIC;
        mem_read_en : out STD_LOGIC;
        mem_ready : in STD_LOGIC
    );
end pcie_dma;

architecture Behavioral of pcie_dma is
    -- DMA state machine
    type dma_state_type is (IDLE, SETUP, READ_SG, TRANSFER, WRITE_SG, DONE, ERROR);
    signal dma_state : dma_state_type := IDLE;

    -- Transfer registers
    signal transfer_count : UNSIGNED (31 downto 0);
    signal current_host_addr : UNSIGNED (63 downto 0);
    signal current_card_addr : UNSIGNED (31 downto 0);
    signal remaining_bytes : UNSIGNED (31 downto 0);

    -- Scatter-gather registers
    signal sg_current_addr : UNSIGNED (63 downto 0);
    signal sg_current_length : UNSIGNED (31 downto 0);
    signal sg_remaining : UNSIGNED (31 downto 0);

    -- Data buffers
    signal rx_buffer : STD_LOGIC_VECTOR (255 downto 0);
    signal tx_buffer : STD_LOGIC_VECTOR (255 downto 0);
    signal buffer_valid : STD_LOGIC;

    -- Performance counters
    signal bytes_transferred : UNSIGNED (63 downto 0);
    signal transfer_rate : UNSIGNED (31 downto 0);

    -- PCIe transaction layer
    type pcie_tlp_type is (MEM_READ, MEM_WRITE, MEM_READ_D, MEM_WRITE_D, CPL, CPL_D);
    signal current_tlp : pcie_tlp_type;

    -- TLP construction
    signal tlp_header : STD_LOGIC_VECTOR (127 downto 0);
    signal tlp_data : STD_LOGIC_VECTOR (255 downto 0);
    signal tlp_valid : STD_LOGIC;

    -- Completion handling
    signal cpl_pending : STD_LOGIC;
    signal cpl_tag : UNSIGNED (7 downto 0);
    signal cpl_addr : UNSIGNED (63 downto 0);

    -- Constants for PCIe Gen4 x16
    constant PCIE_MAX_PAYLOAD : integer := 256;  -- 256 bytes
    constant PCIE_MAX_READ_REQ : integer := 512;  -- 512 bytes
    constant DMA_BURST_SIZE : integer := 64;      -- 64-byte bursts

begin
    -- Main DMA controller state machine
    process(clk, pcie_rst_n)
    begin
        if pcie_rst_n = '0' then
            dma_state <= IDLE;
            dma_done <= '0';
            dma_error <= '0';
            interrupt_req <= '0';
            transfer_count <= (others => '0');
            bytes_transferred <= (others => '0');
            buffer_valid <= '0';
            cpl_pending <= '0';
            mem_write_en <= '0';
            mem_read_en <= '0';
        elsif rising_edge(clk) then
            if reset = '1' then
                dma_state <= IDLE;
                dma_done <= '0';
                dma_error <= '0';
                interrupt_req <= '0';
            else
                case dma_state is
                    when IDLE =>
                        dma_done <= '0';
                        interrupt_req <= '0';

                        if dma_start = '1' then
                            if sg_valid = '1' then
                                -- Scatter-gather mode
                                sg_current_addr <= sg_addr;
                                sg_current_length <= sg_length;
                                sg_remaining <= sg_length;
                                dma_state <= READ_SG;
                            else
                                -- Direct mode
                                current_host_addr <= host_addr;
                                current_card_addr <= card_addr;
                                remaining_bytes <= dma_length;
                                transfer_count <= (others => '0');
                                dma_state <= SETUP;
                            end if;
                        end if;

                    when READ_SG =>
                        -- Read next scatter-gather entry
                        mem_addr <= sg_current_addr(31 downto 0);
                        mem_read_en <= '1';

                        if mem_ready = '1' then
                            mem_read_en <= '0';
                            -- Parse SG entry from memory
                            current_host_addr <= unsigned(mem_data_in(127 downto 64));
                            current_card_addr <= unsigned(mem_data_in(63 downto 32));
                            remaining_bytes <= unsigned(mem_data_in(31 downto 0));
                            transfer_count <= (others => '0');
                            sg_current_addr <= sg_current_addr + 16;  -- Next SG entry
                            sg_remaining <= sg_remaining - 1;
                            dma_state <= SETUP;
                        end if;

                    when SETUP =>
                        -- Setup transfer parameters
                        mem_addr <= current_card_addr;
                        buffer_valid <= '0';

                        if dma_direction = '0' then
                            -- Host to card transfer
                            dma_state <= TRANSFER;
                            -- Initiate PCIe read
                            current_tlp <= MEM_READ;
                            tlp_header <= x"00000000" & -- Format/Type
                                          std_logic_vector(current_host_addr(31 downto 0)) &
                                          std_logic_vector(to_unsigned(PCIE_MAX_READ_REQ, 32));
                        else
                            -- Card to host transfer
                            mem_read_en <= '1';
                            if mem_ready = '1' then
                                mem_read_en <= '0';
                                rx_buffer <= mem_data_in;
                                buffer_valid <= '1';
                                dma_state <= TRANSFER;
                                -- Initiate PCIe write
                                current_tlp <= MEM_WRITE;
                                tlp_header <= x"40000000" & -- Format/Type
                                              std_logic_vector(current_host_addr(31 downto 0)) &
                                              std_logic_vector(to_unsigned(PCIE_MAX_PAYLOAD, 32));
                            end if;
                        end if;

                    when TRANSFER =>
                        -- Execute data transfer
                        if dma_direction = '0' then
                            -- Host to card: wait for PCIe completion
                            if pcie_rx_valid = '1' then
                                rx_buffer <= pcie_rx_data;
                                buffer_valid <= '1';
                                mem_data_out <= pcie_rx_data;
                                mem_addr <= current_card_addr;
                                mem_write_en <= '1';

                                if mem_ready = '1' then
                                    mem_write_en <= '0';
                                    current_card_addr <= current_card_addr + PCIE_MAX_PAYLOAD/32;
                                    current_host_addr <= current_host_addr + PCIE_MAX_PAYLOAD;
                                    remaining_bytes <= remaining_bytes - PCIE_MAX_PAYLOAD;
                                    transfer_count <= transfer_count + 1;
                                    buffer_valid <= '0';

                                    if remaining_bytes <= PCIE_MAX_PAYLOAD then
                                        dma_state <= DONE;
                                    end if;
                                end if;
                            end if;
                        else
                            -- Card to host: send data to host
                            if buffer_valid = '1' then
                                pcie_tx_data <= rx_buffer;
                                pcie_tx_valid <= '1';

                                if pcie_tx_ready = '1' then
                                    pcie_tx_valid <= '0';
                                    current_card_addr <= current_card_addr + PCIE_MAX_PAYLOAD/32;
                                    current_host_addr <= current_host_addr + PCIE_MAX_PAYLOAD;
                                    remaining_bytes <= remaining_bytes - PCIE_MAX_PAYLOAD;
                                    transfer_count <= transfer_count + 1;
                                    buffer_valid <= '0';

                                    if remaining_bytes <= PCIE_MAX_PAYLOAD then
                                        dma_state <= DONE;
                                    else
                                        -- Read next chunk from card memory
                                        mem_addr <= current_card_addr;
                                        mem_read_en <= '1';
                                        if mem_ready = '1' then
                                            mem_read_en <= '0';
                                            rx_buffer <= mem_data_in;
                                            buffer_valid <= '1';
                                        end if;
                                    end if;
                                end if;
                            end if;
                        end if;

                    when WRITE_SG =>
                        -- Update scatter-gather entry (for partial transfers)
                        sg_current_addr <= sg_current_addr - 16;
                        mem_addr <= sg_current_addr(31 downto 0);
                        mem_data_out <= std_logic_vector(current_host_addr) &
                                       std_logic_vector(current_card_addr) &
                                       std_logic_vector(remaining_bytes);
                        mem_write_en <= '1';

                        if mem_ready = '1' then
                            mem_write_en <= '0';
                            sg_remaining <= sg_remaining - 1;

                            if sg_remaining = 0 then
                                dma_state <= DONE;
                            else
                                sg_current_addr <= sg_current_addr + 16;
                                dma_state <= READ_SG;
                            end if;
                        end if;

                    when DONE =>
                        dma_done <= '1';
                        interrupt_req <= '1';
                        bytes_transferred <= bytes_transferred + unsigned(dma_length);

                        -- Clear interrupt after one cycle
                        if interrupt_req = '1' then
                            interrupt_req <= '0';
                            dma_state <= IDLE;
                        end if;

                    when ERROR =>
                        dma_error <= '1';
                        interrupt_req <= '1';

                        if interrupt_req = '1' then
                            interrupt_req <= '0';
                            dma_state <= IDLE;
                        end if;

                    when others =>
                        dma_state <= IDLE;
                end case;
            end if;
        end if;
    end process;

    -- PCIe transaction layer packet construction
    process(clk)
    begin
        if rising_edge(clk) then
            case current_tlp is
                when MEM_READ =>
                    tlp_header <= x"00000000" & -- Memory Read Request
                                  std_logic_vector(current_host_addr(31 downto 0)) &
                                  std_logic_vector(to_unsigned(PCIE_MAX_READ_REQ, 32));
                when MEM_WRITE =>
                    tlp_header <= x"40000000" & -- Memory Write Request
                                  std_logic_vector(current_host_addr(31 downto 0)) &
                                  std_logic_vector(to_unsigned(PCIE_MAX_PAYLOAD, 32));
                when others =>
                    tlp_header <= (others => '0');
            end case;
        end if;
    end process;

    -- Performance monitoring
    process(clk)
    begin
        if rising_edge(clk) then
            if transfer_count > 0 then
                transfer_rate <= bytes_transferred(31 downto 0) / transfer_count;
            end if;
        end if;
    end process;

end Behavioral;
