library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity eth_mac is
    Port (
        -- Clock and reset
        clk : in STD_LOGIC;
        reset : in STD_LOGIC;

        -- GMII interface (to PHY)
        gmii_tx_clk : in STD_LOGIC;
        gmii_rx_clk : in STD_LOGIC;
        gmii_txd : out STD_LOGIC_VECTOR (7 downto 0);
        gmii_tx_en : out STD_LOGIC;
        gmii_tx_er : out STD_LOGIC;
        gmii_rxd : in STD_LOGIC_VECTOR (7 downto 0);
        gmii_rx_dv : in STD_LOGIC;
        gmii_rx_er : in STD_LOGIC;

        -- XGMII interface (10G Ethernet)
        xgmii_tx_clk : in STD_LOGIC;
        xgmii_rx_clk : in STD_LOGIC;
        xgmii_txd : out STD_LOGIC_VECTOR (63 downto 0);
        xgmii_txc : out STD_LOGIC_VECTOR (7 downto 0);
        xgmii_rxd : in STD_LOGIC_VECTOR (63 downto 0);
        xgmii_rxc : in STD_LOGIC_VECTOR (7 downto 0);

        -- MAC interface
        mac_tx_data : in STD_LOGIC_VECTOR (63 downto 0);
        mac_tx_valid : in STD_LOGIC;
        mac_tx_ready : out STD_LOGIC;
        mac_tx_last : in STD_LOGIC;
        mac_tx_error : in STD_LOGIC;

        mac_rx_data : out STD_LOGIC_VECTOR (63 downto 0);
        mac_rx_valid : out STD_LOGIC;
        mac_rx_last : out STD_LOGIC;
        mac_rx_error : out STD_LOGIC;

        -- Control and status
        mac_enable : in STD_LOGIC;
        link_status : out STD_LOGIC;
        speed_mode : in STD_LOGIC_VECTOR (1 downto 0);  -- 00=1G, 01=10G, 10=25G, 11=40G

        -- Statistics
        tx_packets : out UNSIGNED (31 downto 0);
        rx_packets : out UNSIGNED (31 downto 0);
        tx_bytes : out UNSIGNED (63 downto 0);
        rx_bytes : out UNSIGNED (63 downto 0);
        tx_errors : out UNSIGNED (31 downto 0);
        rx_errors : out UNSIGNED (31 downto 0)
    );
end eth_mac;

architecture Behavioral of eth_mac is
    -- MAC state machines
    type tx_state_type is (IDLE, PREAMBLE, SFD, DATA, FCS, INTERFRAME);
    type rx_state_type is (IDLE, PREAMBLE, SFD, DATA, FCS, ERROR);

    signal tx_state : tx_state_type := IDLE;
    signal rx_state : rx_state_type := IDLE;

    -- Transmit buffers
    signal tx_buffer : STD_LOGIC_VECTOR (63 downto 0);
    signal tx_byte_count : UNSIGNED (10 downto 0);
    signal tx_fcs : UNSIGNED (31 downto 0);
    signal tx_crc_valid : STD_LOGIC;

    -- Receive buffers
    signal rx_buffer : STD_LOGIC_VECTOR (63 downto 0);
    signal rx_byte_count : UNSIGNED (10 downto 0);
    signal rx_fcs : UNSIGNED (31 downto 0);
    signal rx_crc_valid : STD_LOGIC;

    -- GMII signals
    signal gmii_tx_data_reg : STD_LOGIC_VECTOR (7 downto 0);
    signal gmii_tx_en_reg : STD_LOGIC;
    signal gmii_tx_er_reg : STD_LOGIC;
    signal gmii_rx_data_reg : STD_LOGIC_VECTOR (7 downto 0);
    signal gmii_rx_dv_reg : STD_LOGIC;
    signal gmii_rx_er_reg : STD_LOGIC;

    -- XGMII signals
    signal xgmii_tx_data_reg : STD_LOGIC_VECTOR (63 downto 0);
    signal xgmii_tx_ctl_reg : STD_LOGIC_VECTOR (7 downto 0);
    signal xgmii_rx_data_reg : STD_LOGIC_VECTOR (63 downto 0);
    signal xgmii_rx_ctl_reg : STD_LOGIC_VECTOR (7 downto 0);

    -- CRC32 calculation
    signal crc32_state : UNSIGNED (31 downto 0);
    signal crc32_input : STD_LOGIC_VECTOR (7 downto 0);

    -- Preamble and SFD
    constant PREAMBLE : STD_LOGIC_VECTOR (55 downto 0) := x"55555555555555";
    constant SFD : STD_LOGIC_VECTOR (7 downto 0) := x"D5";

    -- Speed mode constants
    constant SPEED_1G : STD_LOGIC_VECTOR (1 downto 0) := "00";
    constant SPEED_10G : STD_LOGIC_VECTOR (1 downto 0) := "01";
    constant SPEED_25G : STD_LOGIC_VECTOR (1 downto 0) := "10";
    constant SPEED_40G : STD_LOGIC_VECTOR (1 downto 0) := "11";

    -- Statistics counters
    signal tx_packets_reg : UNSIGNED (31 downto 0);
    signal rx_packets_reg : UNSIGNED (31 downto 0);
    signal tx_bytes_reg : UNSIGNED (63 downto 0);
    signal rx_bytes_reg : UNSIGNED (63 downto 0);
    signal tx_errors_reg : UNSIGNED (31 downto 0);
    signal rx_errors_reg : UNSIGNED (31 downto 0);

    -- CRC32 polynomial (Ethernet)
    constant CRC32_POLY : UNSIGNED (31 downto 0) := x"04C11DB7";

    -- CRC32 calculation function
    function crc32_next(
        crc : UNSIGNED (31 downto 0);
        data : STD_LOGIC_VECTOR (7 downto 0)
    ) return UNSIGNED is
        variable d : UNSIGNED (7 downto 0);
        variable c : UNSIGNED (31 downto 0);
        variable i : integer;
    begin
        d := unsigned(data);
        c := crc xor (d & x"000000");

        for i in 0 to 7 loop
            if c(31) = '1' then
                c := (c(30 downto 0) & '0') xor CRC32_POLY;
            else
                c := c(30 downto 0) & '0';
            end if;
        end loop;

        return c;
    end function;

begin
    -- Transmit state machine
    tx_process: process(clk, reset)
        variable preamble_shift : integer;
    begin
        if reset = '1' then
            tx_state <= IDLE;
            tx_byte_count <= (others => '0');
            tx_fcs <= (others => '0');
            tx_crc_valid <= '0';
            tx_packets_reg <= (others => '0');
            tx_bytes_reg <= (others => '0');
            tx_errors_reg <= (others => '0');
            gmii_tx_en_reg <= '0';
            gmii_tx_er_reg <= '0';
            xgmii_tx_ctl_reg <= (others => '0');
        elsif rising_edge(clk) then
            if mac_enable = '0' then
                tx_state <= IDLE;
                gmii_tx_en_reg <= '0';
                xgmii_tx_ctl_reg <= (others => '0');
            else
                case tx_state is
                    when IDLE =>
                        tx_byte_count <= (others => '0');
                        tx_fcs <= x"FFFFFFFF";  -- CRC32 initial value
                        tx_crc_valid <= '0';

                        if mac_tx_valid = '1' then
                            tx_buffer <= mac_tx_data;
                            tx_state <= DATA;
                            tx_packets_reg <= tx_packets_reg + 1;
                        end if;

                    when DATA =>
                        if mac_tx_ready = '1' then
                            -- Calculate CRC for transmitted data
                            for i in 0 to 7 loop
                                tx_fcs <= crc32_next(tx_fcs, tx_buffer(8*i+7 downto 8*i));
                            end loop;

                            tx_byte_count <= tx_byte_count + 8;
                            tx_bytes_reg <= tx_bytes_reg + 8;

                            if mac_tx_last = '1' then
                                tx_state <= FCS;
                                tx_crc_valid <= '1';
                            elsif mac_tx_valid = '1' then
                                tx_buffer <= mac_tx_data;
                            else
                                tx_state <= ERROR;
                            end if;
                        end if;

                    when FCS =>
                        if tx_crc_valid = '1' then
                            -- Send FCS
                            tx_buffer <= std_logic_vector(tx_fcs);
                            tx_byte_count <= tx_byte_count + 4;
                            tx_bytes_reg <= tx_bytes_reg + 4;
                            tx_crc_valid <= '0';
                            tx_state <= INTERFRAME;
                        end if;

                    when INTERFRAME =>
                        -- Inter-frame gap (96 bit times for 1G, adjusted for higher speeds)
                        if tx_byte_count >= 12 then  -- Minimum frame size
                            tx_state <= IDLE;
                        else
                            tx_errors_reg <= tx_errors_reg + 1;
                            tx_state <= ERROR;
                        end if;

                    when ERROR =>
                        tx_errors_reg <= tx_errors_reg + 1;
                        tx_state <= IDLE;

                    when others =>
                        tx_state <= IDLE;
                end case;
            end if;
        end if;
    end process;

    -- Receive state machine
    rx_process: process(clk, reset)
        variable preamble_count : integer;
    begin
        if reset = '1' then
            rx_state <= IDLE;
            rx_byte_count <= (others => '0');
            rx_fcs <= x"FFFFFFFF";  -- CRC32 initial value
            rx_crc_valid <= '0';
            rx_packets_reg <= (others => '0');
            rx_bytes_reg <= (others => '0');
            rx_errors_reg <= (others => '0');
            mac_rx_valid <= '0';
            mac_rx_last <= '0';
            mac_rx_error <= '0';
        elsif rising_edge(clk) then
            if mac_enable = '0' then
                rx_state <= IDLE;
                mac_rx_valid <= '0';
                mac_rx_last <= '0';
                mac_rx_error <= '0';
            else
                case rx_state is
                    when IDLE =>
                        rx_byte_count <= (others => '0');
                        rx_fcs <= x"FFFFFFFF";
                        rx_crc_valid <= '0';

                        -- Check for preamble start based on speed mode
                        if speed_mode = SPEED_1G then
                            if gmii_rx_dv = '1' and gmii_rx_data_reg = x"55" then
                                preamble_count := 1;
                                rx_state <= PREAMBLE;
                            end if;
                        else
                            -- XGMII preamble detection
                            if xgmii_rx_ctl_reg(0) = '1' and xgmii_rx_data_reg(7 downto 0) = x"55" then
                                preamble_count := 1;
                                rx_state <= PREAMBLE;
                            end if;
                        end if;

                    when PREAMBLE =>
                        if speed_mode = SPEED_1G then
                            if gmii_rx_dv = '1' then
                                if gmii_rx_data_reg = x"55" and preamble_count < 7 then
                                    preamble_count := preamble_count + 1;
                                elsif gmii_rx_data_reg = SFD and preamble_count = 7 then
                                    rx_state <= SFD;
                                else
                                    rx_state <= ERROR;
                                end if;
                            else
                                rx_state <= ERROR;
                            end if;
                        else
                            -- XGMII preamble detection
                            if xgmii_rx_ctl_reg(0) = '1' then
                                if xgmii_rx_data_reg(7 downto 0) = x"55" and preamble_count < 7 then
                                    preamble_count := preamble_count + 1;
                                elsif xgmii_rx_data_reg(7 downto 0) = SFD and preamble_count = 7 then
                                    rx_state <= SFD;
                                else
                                    rx_state <= ERROR;
                                end if;
                            else
                                rx_state <= ERROR;
                            end if;
                        end if;

                    when SFD =>
                        rx_state <= DATA;
                        rx_packets_reg <= rx_packets_reg + 1;

                    when DATA =>
                        if speed_mode = SPEED_1G then
                            if gmii_rx_dv = '1' then
                                -- Calculate CRC for received data
                                rx_fcs <= crc32_next(rx_fcs, gmii_rx_data_reg);
                                rx_byte_count <= rx_byte_count + 1;
                                rx_bytes_reg <= rx_bytes_reg + 1;

                                -- Output data to MAC interface
                                mac_rx_data <= x"00000000000000" & gmii_rx_data_reg;
                                mac_rx_valid <= '1';
                                mac_rx_last <= '0';
                                mac_rx_error <= gmii_rx_er;

                                -- Check for end of frame
                                if gmii_rx_dv = '0' then
                                    rx_state <= FCS;
                                end if;
                            else
                                rx_state <= FCS;
                            end if;
                        else
                            -- XGMII data reception
                            if xgmii_rx_ctl_reg(0) = '1' then
                                -- Calculate CRC for received data
                                for i in 0 to 7 loop
                                    rx_fcs <= crc32_next(rx_fcs, xgmii_rx_data_reg(8*i+7 downto 8*i));
                                end loop;

                                rx_byte_count <= rx_byte_count + 8;
                                rx_bytes_reg <= rx_bytes_reg + 8;

                                -- Output data to MAC interface
                                mac_rx_data <= xgmii_rx_data_reg;
                                mac_rx_valid <= '1';
                                mac_rx_last <= '0';
                                mac_rx_error <= '0';

                                -- Check for end of frame
                                if xgmii_rx_ctl_reg(0) = '0' then
                                    rx_state <= FCS;
                                end if;
                            else
                                rx_state <= FCS;
                            end if;
                        end if;

                    when FCS =>
                        mac_rx_valid <= '0';
                        mac_rx_last <= '1';

                        -- Verify CRC (simplified - would need proper FCS alignment)
                        if rx_fcs = x"2144DF1C" then  -- Expected CRC for valid frame
                            rx_crc_valid <= '1';
                        else
                            rx_errors_reg <= rx_errors_reg + 1;
                            mac_rx_error <= '1';
                        end if;

                        rx_state <= IDLE;

                    when ERROR =>
                        rx_errors_reg <= rx_errors_reg + 1;
                        mac_rx_error <= '1';
                        mac_rx_valid <= '0';
                        mac_rx_last <= '1';
                        rx_state <= IDLE;

                    when others =>
                        rx_state <= IDLE;
                end case;
            end if;
        end if;
    end process;

    -- GMII interface (1G Ethernet)
    gmii_process: process(clk)
    begin
        if rising_edge(clk) then
            if speed_mode = SPEED_1G then
                -- Transmit
                gmii_tx_data_reg <= tx_buffer(7 downto 0);
                gmii_tx_en_reg <= mac_tx_valid;
                gmii_tx_er_reg <= mac_tx_error;

                -- Receive
                gmii_rx_data_reg <= gmii_rxd;
                gmii_rx_dv_reg <= gmii_rx_dv;
                gmii_rx_er_reg <= gmii_rx_er;
            else
                -- Disable GMII for higher speeds
                gmii_tx_data_reg <= (others => '0');
                gmii_tx_en_reg <= '0';
                gmii_tx_er_reg <= '0';
            end if;
        end if;
    end process;

    -- XGMII interface (10G/25G/40G Ethernet)
    xgmii_process: process(clk)
    begin
        if rising_edge(clk) then
            if speed_mode /= SPEED_1G then
                -- Transmit
                xgmii_tx_data_reg <= mac_tx_data;
                xgmii_tx_ctl_reg <= (others => mac_tx_valid);

                -- Receive
                xgmii_rx_data_reg <= xgmii_rxd;
                xgmii_rx_ctl_reg <= xgmii_rxc;
            else
                -- Disable XGMII for 1G
                xgmii_tx_data_reg <= (others => '0');
                xgmii_tx_ctl_reg <= (others => '0');
            end if;
        end if;
    end process;

    -- Output assignments
    gmii_txd <= gmii_tx_data_reg;
    gmii_tx_en <= gmii_tx_en_reg;
    gmii_tx_er <= gmii_tx_er_reg;

    xgmii_txd <= xgmii_tx_data_reg;
    xgmii_txc <= xgmii_tx_ctl_reg;

    mac_tx_ready <= '1' when tx_state = DATA else '0';

    tx_packets <= tx_packets_reg;
    rx_packets <= rx_packets_reg;
    tx_bytes <= tx_bytes_reg;
    rx_bytes <= rx_bytes_reg;
    tx_errors <= tx_errors_reg;
    rx_errors <= rx_errors_reg;

    -- Link status (simplified)
    link_status <= '1' when mac_enable = '1' else '0';

end Behavioral;
