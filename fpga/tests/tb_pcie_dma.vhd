library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity tb_pcie_dma is
end tb_pcie_dma;

architecture Behavioral of tb_pcie_dma is
    -- Component Declaration
    component pcie_dma
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

    -- Signals
    signal clk : STD_LOGIC := '0';
    signal reset : STD_LOGIC := '0';
    signal pcie_clk : STD_LOGIC := '0';
    signal pcie_rst_n : STD_LOGIC := '1';
    signal pcie_rx_data : STD_LOGIC_VECTOR (255 downto 0) := (others => '0');
    signal pcie_rx_valid : STD_LOGIC := '0';
    signal pcie_tx_data : STD_LOGIC_VECTOR (255 downto 0);
    signal pcie_tx_valid : STD_LOGIC;
    signal pcie_tx_ready : STD_LOGIC := '1';
    signal dma_start : STD_LOGIC := '0';
    signal dma_direction : STD_LOGIC := '0';
    signal dma_length : UNSIGNED (31 downto 0) := (others => '0');
    signal host_addr : UNSIGNED (63 downto 0) := (others => '0');
    signal card_addr : UNSIGNED (31 downto 0) := (others => '0');
    signal sg_addr : UNSIGNED (63 downto 0) := (others => '0');
    signal sg_length : UNSIGNED (31 downto 0) := (others => '0');
    signal sg_valid : STD_LOGIC := '0';
    signal dma_done : STD_LOGIC;
    signal dma_error : STD_LOGIC;
    signal interrupt_req : STD_LOGIC;
    signal mem_addr : UNSIGNED (31 downto 0);
    signal mem_data_in : STD_LOGIC_VECTOR (255 downto 0) := (others => '0');
    signal mem_data_out : STD_LOGIC_VECTOR (255 downto 0);
    signal mem_write_en : STD_LOGIC;
    signal mem_read_en : STD_LOGIC;
    signal mem_ready : STD_LOGIC := '1';

    -- Clock period definitions
    constant clk_period : time := 5 ns; -- 200 MHz
    constant pcie_clk_period : time := 4 ns; -- 250 MHz

begin
    -- Instantiate the Unit Under Test (UUT)
    uut: pcie_dma Port Map (
        clk => clk,
        reset => reset,
        pcie_clk => pcie_clk,
        pcie_rst_n => pcie_rst_n,
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

    -- Clock process definitions
    clk_process :process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

    pcie_clk_process :process
    begin
        pcie_clk <= '0';
        wait for pcie_clk_period/2;
        pcie_clk <= '1';
        wait for pcie_clk_period/2;
    end process;

    -- Stimulus process
    stim_proc: process
    begin		
        -- Reset
        pcie_rst_n <= '0';
        reset <= '1';
        wait for 20 ns;
        pcie_rst_n <= '1';
        reset <= '0';
        wait for clk_period * 5;

        -- Test Case 1: Start Host to Card DMA (Direct Mode)
        host_addr <= x"00000000_12345678";
        card_addr <= x"A0000000";
        dma_length <= x"00000100"; -- 256 bytes
        dma_direction <= '0';
        dma_start <= '1';
        wait for clk_period;
        dma_start <= '0';

        -- Simulate PCIe RX data arriving
        wait for clk_period * 5;
        pcie_rx_data <= (others => 'A');
        pcie_rx_valid <= '1';
        wait for clk_period;
        pcie_rx_valid <= '0';

        -- Wait for completion
        wait until dma_done = '1';
        wait for clk_period * 10;

        -- Test Case 2: Start Card to Host DMA
        host_addr <= x"00000000_87654321";
        card_addr <= x"B0000000";
        dma_length <= x"00000100";
        dma_direction <= '1';
        dma_start <= '1';
        wait for clk_period;
        dma_start <= '0';

        -- Simulate memory ready
        mem_data_in <= (others => 'B');
        
        wait until dma_done = '1';

        wait for 100 ns;
        wait;
    end process;

end Behavioral;
