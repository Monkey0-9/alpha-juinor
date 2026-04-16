library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity tb_order_book is
end tb_order_book;

architecture Behavioral of tb_order_book is
    -- Component Declaration
    component order_book
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

    -- Signals
    signal clk : STD_LOGIC := '0';
    signal reset : STD_LOGIC := '0';
    signal order_in : STD_LOGIC_VECTOR (127 downto 0) := (others => '0');
    signal order_valid : STD_LOGIC := '0';
    signal best_bid : STD_LOGIC_VECTOR (63 downto 0);
    signal best_ask : STD_LOGIC_VECTOR (63 downto 0);
    signal trade_out : STD_LOGIC_VECTOR (127 downto 0);
    signal trade_valid : STD_LOGIC;
    signal book_full : STD_LOGIC;

    -- Clock period definitions
    constant clk_period : time := 5 ns; -- 200 MHz

begin
    -- Instantiate the Unit Under Test (UUT)
    uut: order_book Port Map (
        clk => clk,
        reset => reset,
        order_in => order_in,
        order_valid => order_valid,
        best_bid => best_bid,
        best_ask => best_ask,
        trade_out => trade_out,
        trade_valid => trade_valid,
        book_full => book_full
    );

    -- Clock process definitions
    clk_process :process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

    -- Stimulus process
    stim_proc: process
        procedure send_order(price: integer; qty: integer; side: std_logic; id: integer) is
        begin
            order_in(127 downto 96) <= std_logic_vector(to_unsigned(price, 32));
            order_in(95 downto 64) <= std_logic_vector(to_unsigned(qty, 32));
            order_in(63) <= side;
            order_in(62 downto 56) <= (others => '0');
            order_in(55 downto 48) <= x"AA"; -- Symbol
            order_in(47 downto 0) <= std_logic_vector(to_unsigned(id, 48));
            order_valid <= '1';
            wait for clk_period;
            order_valid <= '0';
            wait for clk_period;
        end procedure;

    begin		
        -- Reset
        reset <= '1';
        wait for 20 ns;
        reset <= '0';
        wait for clk_period * 5;

        -- Test Case 1: Add a Bid
        send_order(150, 100, '0', 1);
        wait for clk_period * 2;

        -- Test Case 2: Add an Ask (no match)
        send_order(155, 50, '1', 2);
        wait for clk_period * 2;

        -- Test Case 3: Add a matching Buy order
        send_order(156, 30, '0', 3); -- Should match against 155 ask
        wait for clk_period * 2;

        -- Test Case 4: Complex matching
        send_order(149, 200, '1', 4); -- Should match against 150 bid
        
        wait for 100 ns;
        -- Use finish in modern VHDL or just stop toggle
        wait;
    end process;

end Behavioral;
