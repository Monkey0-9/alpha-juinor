library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity matching_engine is
    Port ( clk : in STD_LOGIC;
           reset : in STD_LOGIC;
           new_order : in STD_LOGIC;
           order_side : in STD_LOGIC; -- 0 for Buy, 1 for Sell
           order_price : in UNSIGNED (31 downto 0);
           order_qty : in UNSIGNED (31 downto 0);
           match_occurred : out STD_LOGIC;
           match_price : out UNSIGNED (31 downto 0);
           match_qty : out UNSIGNED (31 downto 0));
end matching_engine;

architecture Behavioral of matching_engine is
    -- Simplified hardware matching logic
    type price_level is record
        price : UNSIGNED (31 downto 0);
        qty   : UNSIGNED (31 downto 0);
        valid : STD_LOGIC;
    end record;

    type book_side is array (0 to 15) of price_level;
    signal bid_book : book_side := (others => (price => (others => '0'), qty => (others => '0'), valid => '0'));
    signal ask_book : book_side := (others => (price => (others => '0'), qty => (others => '0'), valid => '0'));

begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                match_occurred <= '0';
            elsif new_order = '1' then
                -- Matching logic: If Buy, check Ask Book
                if order_side = '0' then
                    if ask_book(0).valid = '1' and order_price >= ask_book(0).price then
                        match_occurred <= '1';
                        match_price <= ask_book(0).price;
                        match_qty <= order_qty; -- Simplified
                    end if;
                else
                    -- If Sell, check Bid Book
                    if bid_book(0).valid = '1' and order_price <= bid_book(0).price then
                        match_occurred <= '1';
                        match_price <= bid_book(0).price;
                        match_qty <= order_qty;
                    end if;
                end if;
            else
                match_occurred <= '0';
            end if;
        end if;
    end process;
end Behavioral;
