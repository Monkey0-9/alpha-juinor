library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity top_level is
    Port ( 
        -- System clocks
        sys_clk : in STD_LOGIC;  -- 100MHz system clock
        pcie_clk : in STD_LOGIC;  -- PCIe reference clock
        eth_clk : in STD_LOGIC;   -- Ethernet reference clock
        
        -- Reset
        reset_n : in STD_LOGIC;
        
        -- PCIe interface (Gen4 x16)
        pcie_rx_p : in STD_LOGIC_VECTOR (15 downto 0);
        pcie_rx_n : in STD_LOGIC_VECTOR (15 downto 0);
        pcie_tx_p : out STD_LOGIC_VECTOR (15 downto 0);
        pcie_tx_n : out STD_LOGIC_VECTOR (15 downto 0);
        
        -- Ethernet interface (10G/25G)
        eth_rx_p : in STD_LOGIC_VECTOR (3 downto 0);
        eth_rx_n : in STD_LOGIC_VECTOR (3 downto 0);
        eth_tx_p : out STD_LOGIC_VECTOR (3 downto 0);
        eth_tx_n : out STD_LOGIC_VECTOR (3 downto 0);
        
        -- Status LEDs
        led_link : out STD_LOGIC;
        led_activity : out STD_LOGIC;
        led_error : out STD_LOGIC;
        
        -- Debug interface
        debug_uart_tx : out STD_LOGIC;
        debug_uart_rx : in STD_LOGIC
    );
end top_level;

architecture Behavioral of top_level is
    -- Component declarations
    component order_book is
        Port ( 
            clk : in STD_LOGIC;
            reset : in STD_LOGIC;
            order_in : in STD_LOGIC_VECTOR (127 downto 0);
            order_valid : in STD_LOGIC;
            best_bid : out STD_LOGIC_VECTOR (63 downto 0);
            best_ask : out STD_LOGIC_VECTOR (63 downto 0);
            trade_out : out STD_LOGIC_VECTOR (127 downto 0);
            trade_valid : out STD_LOGIC;
            book_full : out STD_LOGIC
        );
    end component;
    
    component matching_engine is
        Port ( 
            clk : in STD_LOGIC;
            reset : in STD_LOGIC;
            new_order : in STD_LOGIC;
            order_side : in STD_LOGIC;
            order_price : in UNSIGNED (31 downto 0);
            order_qty : in UNSIGNED (31 downto 0);
            order_id : in UNSIGNED (47 downto 0);
            best_bid_price : in UNSIGNED (31 downto 0);
            best_bid_qty : in UNSIGNED (31 downto 0);
            best_ask_price : in UNSIGNED (31 downto 0);
            best_ask_qty : in UNSIGNED (31 downto 0);
            match_occurred : out STD_LOGIC;
            match_price : out UNSIGNED (31 downto 0);
            match_qty : out UNSIGNED (31 downto 0);
            aggressor_id : out UNSIGNED (47 downto 0);
            passive_id : out UNSIGNED (47 downto 0);
            update_bid : out STD_LOGIC;
            update_ask : out STD_LOGIC;
            new_best_bid : out UNSIGNED (31 downto 0);
            new_best_bid_qty : out UNSIGNED (31 downto 0);
            new_best_ask : out UNSIGNED (31 downto 0);
            new_best_ask_qty : out UNSIGNED (31 downto 0)
        );
    end component;
    
    component pcie_dma is
        Port ( 
            clk : in STD_LOGIC;
            reset : in STD_LOGIC;
            pcie_clk : in STD_LOGIC;
            pcie_rst_n : in STD_LOGIC;
            pcie_rx_data : in STD_LOGIC_VECTOR (255 downto 0);
            pcie_rx_valid : in STD_LOGIC;
            pcie_tx_data : out STD_LOGIC_VECTOR (255 downto 0);
            pcie_tx_valid : out STD_LOGIC;
            pcie_tx_ready : in STD_LOGIC;
            dma_start : in STD_LOGIC;
            dma_direction : in STD_LOGIC;
            dma_length : in UNSIGNED (31 downto 0);
            host_addr : in UNSIGNED (63 downto 0);
            card_addr : in UNSIGNED (31 downto 0);
            sg_addr : in UNSIGNED (63 downto 0);
            sg_length : in UNSIGNED (31 downto 0);
            sg_valid : in STD_LOGIC;
            dma_done : out STD_LOGIC;
            dma_error : out STD_LOGIC;
            interrupt_req : out STD_LOGIC;
            mem_addr : out UNSIGNED (31 downto 0);
            mem_data_in : in STD_LOGIC_VECTOR (255 downto 0);
            mem_data_out : out STD_LOGIC_VECTOR (255 downto 0);
            mem_write_en : out STD_LOGIC;
            mem_read_en : out STD_LOGIC;
            mem_ready : in STD_LOGIC
        );
    end component;
    
    component eth_mac is
        Port ( 
            clk : in STD_LOGIC;
            reset : in STD_LOGIC;
            gmii_tx_clk : in STD_LOGIC;
            gmii_rx_clk : in STD_LOGIC;
            gmii_txd : out STD_LOGIC_VECTOR (7 downto 0);
            gmii_tx_en : out STD_LOGIC;
            gmii_tx_er : out STD_LOGIC;
            gmii_rxd : in STD_LOGIC_VECTOR (7 downto 0);
            gmii_rx_dv : in STD_LOGIC;
            gmii_rx_er : in STD_LOGIC;
            xgmii_tx_clk : in STD_LOGIC;
            xgmii_rx_clk : in STD_LOGIC;
            xgmii_txd : out STD_LOGIC_VECTOR (63 downto 0);
            xgmii_txc : out STD_LOGIC_VECTOR (7 downto 0);
            xgmii_rxd : in STD_LOGIC_VECTOR (63 downto 0);
            xgmii_rxc : in STD_LOGIC_VECTOR (7 downto 0);
            mac_tx_data : in STD_LOGIC_VECTOR (63 downto 0);
            mac_tx_valid : in STD_LOGIC;
            mac_tx_ready : out STD_LOGIC;
            mac_tx_last : in STD_LOGIC;
            mac_tx_error : in STD_LOGIC;
            mac_rx_data : out STD_LOGIC_VECTOR (63 downto 0);
            mac_rx_valid : out STD_LOGIC;
            mac_rx_last : out STD_LOGIC;
            mac_rx_error : out STD_LOGIC;
            mac_enable : in STD_LOGIC;
            link_status : out STD_LOGIC;
            speed_mode : in STD_LOGIC_VECTOR (1 downto 0);
            tx_packets : out UNSIGNED (31 downto 0);
            rx_packets : out UNSIGNED (31 downto 0);
            tx_bytes : out UNSIGNED (63 downto 0);
            rx_bytes : out UNSIGNED (63 downto 0);
            tx_errors : out UNSIGNED (31 downto 0);
            rx_errors : out UNSIGNED (31 downto 0)
        );
    end component;
    
    -- Clock management
    signal core_clk : STD_LOGIC;
    signal eth_core_clk : STD_LOGIC;
    signal pcie_core_clk : STD_LOGIC;
    
    -- Reset synchronization
    signal core_reset : STD_LOGIC;
    signal eth_reset : STD_LOGIC;
    signal pcie_reset : STD_LOGIC;
    
    -- Order book signals
    signal order_in : STD_LOGIC_VECTOR (127 downto 0);
    signal order_valid : STD_LOGIC;
    signal best_bid : STD_LOGIC_VECTOR (63 downto 0);
    signal best_ask : STD_LOGIC_VECTOR (63 downto 0);
    signal trade_out : STD_LOGIC_VECTOR (127 downto 0);
    signal trade_valid : STD_LOGIC;
    signal book_full : STD_LOGIC;
    
    -- Matching engine signals
    signal match_occurred : STD_LOGIC;
    signal match_price : UNSIGNED (31 downto 0);
    signal match_qty : UNSIGNED (31 downto 0);
    signal aggressor_id : UNSIGNED (47 downto 0);
    signal passive_id : UNSIGNED (47 downto 0);
    signal update_bid : STD_LOGIC;
    signal update_ask : STD_LOGIC;
    signal new_best_bid : UNSIGNED (31 downto 0);
    signal new_best_bid_qty : UNSIGNED (31 downto 0);
    signal new_best_ask : UNSIGNED (31 downto 0);
    signal new_best_ask_qty : UNSIGNED (31 downto 0);
    
    -- PCIe DMA signals
    signal pcie_rx_data : STD_LOGIC_VECTOR (255 downto 0);
    signal pcie_rx_valid : STD_LOGIC;
    signal pcie_tx_data : STD_LOGIC_VECTOR (255 downto 0);
    signal pcie_tx_valid : STD_LOGIC;
    signal pcie_tx_ready : STD_LOGIC;
    signal dma_start : STD_LOGIC;
    signal dma_direction : STD_LOGIC;
    signal dma_length : UNSIGNED (31 downto 0);
    signal host_addr : UNSIGNED (63 downto 0);
    signal card_addr : UNSIGNED (31 downto 0);
    signal sg_addr : UNSIGNED (63 downto 0);
    signal sg_length : UNSIGNED (31 downto 0);
    signal sg_valid : STD_LOGIC;
    signal dma_done : STD_LOGIC;
    signal dma_error : STD_LOGIC;
    signal interrupt_req : STD_LOGIC;
    signal mem_addr : UNSIGNED (31 downto 0);
    signal mem_data_in : STD_LOGIC_VECTOR (255 downto 0);
    signal mem_data_out : STD_LOGIC_VECTOR (255 downto 0);
    signal mem_write_en : STD_LOGIC;
    signal mem_read_en : STD_LOGIC;
    signal mem_ready : STD_LOGIC;
    
    -- Ethernet MAC signals
    signal mac_tx_data : STD_LOGIC_VECTOR (63 downto 0);
    signal mac_tx_valid : STD_LOGIC;
    signal mac_tx_ready : STD_LOGIC;
    signal mac_tx_last : STD_LOGIC;
    signal mac_tx_error : STD_LOGIC;
    signal mac_rx_data : STD_LOGIC_VECTOR (63 downto 0);
    signal mac_rx_valid : STD_LOGIC;
    signal mac_rx_last : STD_LOGIC;
    signal mac_rx_error : STD_LOGIC;
    signal mac_enable : STD_LOGIC;
    signal link_status : STD_LOGIC;
    signal speed_mode : STD_LOGIC_VECTOR (1 downto 0);
    signal tx_packets : UNSIGNED (31 downto 0);
    signal rx_packets : UNSIGNED (31 downto 0);
    signal tx_bytes : UNSIGNED (63 downto 0);
    signal rx_bytes : UNSIGNED (63 downto 0);
    signal tx_errors : UNSIGNED (31 downto 0);
    signal rx_errors : UNSIGNED (31 downto 0);
    
    -- Internal memory (BRAM)
    type bram_array is array (0 to 1023) of STD_LOGIC_VECTOR (255 downto 0);
    signal bram : bram_array;
    
    -- Control registers
    signal control_reg : STD_LOGIC_VECTOR (31 downto 0);
    signal status_reg : STD_LOGIC_VECTOR (31 downto 0);
    signal latency_counter : UNSIGNED (31 downto 0);
    signal performance_counter : UNSIGNED (63 downto 0);
    
    -- Debug signals
    signal debug_data : STD_LOGIC_VECTOR (7 downto 0);
    signal debug_valid : STD_LOGIC;
    
begin
    -- Clock and reset management
    clock_manager: process(sys_clk, reset_n)
    begin
        if reset_n = '0' then
            core_clk <= '0';
            eth_core_clk <= '0';
            pcie_core_clk <= '0';
            core_reset <= '1';
            eth_reset <= '1';
            pcie_reset <= '1';
        elsif rising_edge(sys_clk) then
            core_clk <= sys_clk;
            eth_core_clk <= eth_clk;
            pcie_core_clk <= pcie_clk;
            core_reset <= not reset_n;
            eth_reset <= not reset_n;
            pcie_reset <= not reset_n;
        end if;
    end process;
    
    -- Order Book instance
    order_book_inst: order_book
        port map (
            clk => core_clk,
            reset => core_reset,
            order_in => order_in,
            order_valid => order_valid,
            best_bid => best_bid,
            best_ask => best_ask,
            trade_out => trade_out,
            trade_valid => trade_valid,
            book_full => book_full
        );
    
    -- Matching Engine instance
    matching_engine_inst: matching_engine
        port map (
            clk => core_clk,
            reset => core_reset,
            new_order => order_valid,
            order_side => order_in(63),
            order_price => unsigned(order_in(127 downto 96)),
            order_qty => unsigned(order_in(95 downto 64)),
            order_id => unsigned(order_in(47 downto 0)),
            best_bid_price => unsigned(best_bid(63 downto 32)),
            best_bid_qty => unsigned(best_bid(31 downto 0)),
            best_ask_price => unsigned(best_ask(63 downto 32)),
            best_ask_qty => unsigned(best_ask(31 downto 0)),
            match_occurred => match_occurred,
            match_price => match_price,
            match_qty => match_qty,
            aggressor_id => aggressor_id,
            passive_id => passive_id,
            update_bid => update_bid,
            update_ask => update_ask,
            new_best_bid => new_best_bid,
            new_best_bid_qty => new_best_bid_qty,
            new_best_ask => new_best_ask,
            new_best_ask_qty => new_best_ask_qty
        );
    
    -- PCIe DMA instance
    pcie_dma_inst: pcie_dma
        port map (
            clk => core_clk,
            reset => core_reset,
            pcie_clk => pcie_core_clk,
            pcie_rst_n => reset_n,
            pcie_rx_data => pcie_rx_data,
            pcie_rx_valid => pcie_rx_valid,
            pcie_tx_data => pcie_tx_data,
            pcie_tx_valid => pcie_tx_valid,
            pcie_tx_ready => pcie_tx_ready,
            dma_start => dma_start,
            dma_direction => dma_direction,
            dma_length => dma_length,
            host_addr => host_addr,
            card_addr => card_addr,
            sg_addr => sg_addr,
            sg_length => sg_length,
            sg_valid => sg_valid,
            dma_done => dma_done,
            dma_error => dma_error,
            interrupt_req => interrupt_req,
            mem_addr => mem_addr,
            mem_data_in => mem_data_in,
            mem_data_out => mem_data_out,
            mem_write_en => mem_write_en,
            mem_read_en => mem_read_en,
            mem_ready => mem_ready
        );
    
    -- Ethernet MAC instance
    eth_mac_inst: eth_mac
        port map (
            clk => eth_core_clk,
            reset => eth_reset,
            gmii_tx_clk => eth_clk,
            gmii_rx_clk => eth_clk,
            gmii_txd => open,  -- Not used for 10G+
            gmii_tx_en => open,
            gmii_tx_er => open,
            gmii_rxd => (others => '0'),
            gmii_rx_dv => '0',
            gmii_rx_er => '0',
            xgmii_tx_clk => eth_clk,
            xgmii_rx_clk => eth_clk,
            xgmii_txd => eth_tx_data,
            xgmii_txc => open,
            xgmii_rxd => eth_rx_data,
            xgmii_rxc => open,
            mac_tx_data => mac_tx_data,
            mac_tx_valid => mac_tx_valid,
            mac_tx_ready => mac_tx_ready,
            mac_tx_last => mac_tx_last,
            mac_tx_error => mac_tx_error,
            mac_rx_data => mac_rx_data,
            mac_rx_valid => mac_rx_valid,
            mac_rx_last => mac_rx_last,
            mac_rx_error => mac_rx_error,
            mac_enable => mac_enable,
            link_status => link_status,
            speed_mode => speed_mode,
            tx_packets => tx_packets,
            rx_packets => rx_packets,
            tx_bytes => tx_bytes,
            rx_bytes => rx_bytes,
            tx_errors => tx_errors,
            rx_errors => rx_errors
        );
    
    -- Internal BRAM interface
    bram_interface: process(core_clk)
        variable bram_addr : integer;
    begin
        if rising_edge(core_clk) then
            if mem_write_en = '1' then
                bram_addr := to_integer(mem_addr(9 downto 0));
                bram(bram_addr) <= mem_data_out;
            end if;
            
            if mem_read_en = '1' then
                bram_addr := to_integer(mem_addr(9 downto 0));
                mem_data_in <= bram(bram_addr);
            else
                mem_data_in <= (others => '0');
            end if;
            
            mem_ready <= '1';
        end if;
    end process;
    
    -- Main control logic
    main_control: process(core_clk)
        variable order_counter : integer := 0;
        variable trade_counter : integer := 0;
    begin
        if rising_edge(core_clk) then
            if core_reset = '1' then
                order_valid <= '0';
                order_in <= (others => '0');
                dma_start <= '0';
                dma_direction <= '0';
                dma_length <= (others => '0');
                host_addr <= (others => '0');
                card_addr <= (others => '0');
                mac_enable <= '0';
                speed_mode <= "01";  -- 10G mode
                control_reg <= (others => '0');
                status_reg <= (others => '0');
                latency_counter <= (others => '0');
                performance_counter <= (others => '0');
                order_counter := 0;
                trade_counter := 0;
            else
                -- Default values
                order_valid <= '0';
                dma_start <= '0';
                mac_enable <= '1';
                
                -- Process incoming orders from PCIe
                if pcie_rx_valid = '1' then
                    order_in <= pcie_rx_data(127 downto 0);
                    order_valid <= '1';
                    order_counter := order_counter + 1;
                    latency_counter <= latency_counter + 1;
                end if;
                
                -- Handle matches and send trades back
                if match_occurred = '1' then
                    trade_counter := trade_counter + 1;
                    performance_counter <= performance_counter + 1;
                    
                    -- Send trade via PCIe
                    mem_data_out <= std_logic_vector(match_price) & 
                                   std_logic_vector(match_qty) &
                                   std_logic_vector(aggressor_id) &
                                   std_logic_vector(passive_id);
                    mem_addr <= x"00000100";  -- Trade buffer address
                    mem_write_en <= '1';
                    
                    -- Initiate DMA transfer
                    dma_start <= '1';
                    dma_direction <= '1';  -- Card to host
                    dma_length <= x"00000020";  -- 32 bytes
                    host_addr <= x"0000000010000000";  -- Host buffer
                    card_addr <= x"00000100";  -- Card buffer
                else
                    mem_write_en <= '0';
                end if;
                
                -- Update status register
                status_reg(0) <= link_status;
                status_reg(1) <= book_full;
                status_reg(2) <= match_occurred;
                status_reg(3) <= dma_error;
                status_reg(4) <= mac_rx_error;
                status_reg(5) <= '1' when tx_packets > 0 else '0';
                status_reg(6) <= '1' when rx_packets > 0 else '0';
                status_reg(7) <= interrupt_req;
                status_reg(15 downto 8) <= std_logic_vector(latency_counter(7 downto 0));
                status_reg(31 downto 16) <= std_logic_vector(to_unsigned(order_counter, 16));
            end if;
        end if;
    end process;
    
    -- Ethernet data processing (simplified)
    eth_processing: process(eth_core_clk)
    begin
        if rising_edge(eth_core_clk) then
            if mac_rx_valid = '1' then
                -- Process received market data
                -- For now, just forward to order book
                order_in <= mac_rx_data(127 downto 0);
                order_valid <= '1';
            else
                order_valid <= '0';
            end if;
            
            -- Transmit market data updates
            if trade_valid = '1' then
                mac_tx_data <= trade_out;
                mac_tx_valid <= '1';
                mac_tx_last <= '1';
                mac_tx_error <= '0';
            else
                mac_tx_valid <= '0';
                mac_tx_last <= '0';
            end if;
        end if;
    end process;
    
    -- Status LEDs
    led_link <= link_status;
    led_activity <= '1' when (match_occurred = '1' or trade_valid = '1') else '0';
    led_error <= dma_error or mac_rx_error or book_full;
    
    -- Debug UART (simplified)
    debug_uart: process(core_clk)
        variable debug_counter : integer := 0;
    begin
        if rising_edge(core_clk) then
            if debug_counter < 1000000 then
                debug_counter := debug_counter + 1;
                debug_uart_tx <= '1';  -- Idle
            else
                debug_counter := 0;
                -- Send status byte
                debug_uart_tx <= status_reg(0);
            end if;
        end if;
    end process;
    
    -- Differential output buffers (would need actual IBUFDS/OBUFDS primitives)
    -- For simulation purposes, just assign single-ended signals
    
end Behavioral;
