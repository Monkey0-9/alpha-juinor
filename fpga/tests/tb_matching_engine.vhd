-- Testbench for matching_engine
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity tb_matching_engine is
end tb_matching_engine;

architecture Behavioral of tb_matching_engine is
    -- Component Declaration
    component matching_engine
        Port (
            clk : in STD_LOGIC;
            reset : in STD_LOGIC;
            order_in : in STD_LOGIC_VECTOR (127 downto 0);
            order_valid : in STD_LOGIC;
            trade_out : out STD_LOGIC_VECTOR (127 downto 0);
            trade_valid : out STD_LOGIC
        );
    end component;

    -- Signals
    signal clk : STD_LOGIC := '0';
    signal reset : STD_LOGIC := '0';
    signal order_in : STD_LOGIC_VECTOR (127 downto 0) := (others => '0');
    signal order_valid : STD_LOGIC := '0';
    signal trade_out : STD_LOGIC_VECTOR (127 downto 0);
    signal trade_valid : STD_LOGIC;

    constant clk_period : time := 10 ns;

begin
    -- Instantiate UUT
    uut: matching_engine Port Map (
        clk => clk,
        reset => reset,
        order_in => order_in,
        order_valid => order_valid,
        trade_out => trade_out,
        trade_valid => trade_valid
    );

    -- Clock process
    clk_process :process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

    -- Stimulus process
    stim_proc: process
    begin
        reset <= '1';
        wait for 20 ns;
        reset <= '0';
        wait for 20 ns;

        -- Test Case 1: Send Bid Order
        -- [127:96]=price(100), [95:64]=qty(10), [63]=side(0), [55:48]=sym(1), [47:0]=id(1)
        order_in <= x"000000640000000A0001000000000001";
        order_valid <= '1';
        wait for clk_period;
        order_valid <= '0';
        wait for 20 ns;

        -- Test Case 2: Send Matching Ask Order
        -- [127:96]=price(100), [95:64]=qty(10), [63]=side(1), [55:48]=sym(1), [47:0]=id(2)
        order_in <= x"000000640000000A8001000000000002";
        order_valid <= '1';
        wait for clk_period;
        order_valid <= '0';

        wait for 100 ns;
        assert trade_valid = '1' report "Match failed!" severity error;

        wait;
    end process;
end Behavioral;
