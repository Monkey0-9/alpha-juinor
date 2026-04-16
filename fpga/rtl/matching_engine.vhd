library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity matching_engine is
    Port (
        clk : in STD_LOGIC;
        reset : in STD_LOGIC;
        -- Order input
        new_order : in STD_LOGIC;
        order_side : in STD_LOGIC; -- 0 for Buy, 1 for Sell
        order_price : in UNSIGNED (31 downto 0);
        order_qty : in UNSIGNED (31 downto 0);
        order_id : in UNSIGNED (47 downto 0);
        -- Book state input
        best_bid_price : in UNSIGNED (31 downto 0);
        best_bid_qty : in UNSIGNED (31 downto 0);
        best_ask_price : in UNSIGNED (31 downto 0);
        best_ask_qty : in UNSIGNED (31 downto 0);
        -- Match outputs
        match_occurred : out STD_LOGIC;
        match_price : out UNSIGNED (31 downto 0);
        match_qty : out UNSIGNED (31 downto 0);
        aggressor_id : out UNSIGNED (47 downto 0);
        passive_id : out UNSIGNED (47 downto 0);
        -- Book updates
        update_bid : out STD_LOGIC;
        update_ask : out STD_LOGIC;
        new_best_bid : out UNSIGNED (31 downto 0);
        new_best_bid_qty : out UNSIGNED (31 downto 0);
        new_best_ask : out UNSIGNED (31 downto 0);
        new_best_ask_qty : out UNSIGNED (31 downto 0)
    );
end matching_engine;

architecture Behavioral of matching_engine is
    -- Matching state machine
    type match_state_type is (IDLE, CHECK_MATCH, EXECUTE_MATCH, UPDATE_BOOK);
    signal match_state : match_state_type := IDLE;

    -- Match calculation registers
    signal remaining_qty : UNSIGNED (31 downto 0);
    signal match_price_reg : UNSIGNED (31 downto 0);
    signal aggressor_id_reg : UNSIGNED (47 downto 0);
    signal passive_id_reg : UNSIGNED (47 downto 0);

    -- Price-time priority logic
    function can_match(
        side : STD_LOGIC;
        order_price : UNSIGNED (31 downto 0);
        best_price : UNSIGNED (31 downto 0);
        best_qty : UNSIGNED (31 downto 0)
    ) return BOOLEAN is
    begin
        if side = '0' then  -- Buy order
            return (best_qty > 0 and order_price >= best_price);
        else  -- Sell order
            return (best_qty > 0 and order_price <= best_price);
        end if;
    end function;

    function calculate_match_qty(
        order_qty : UNSIGNED (31 downto 0);
        best_qty : UNSIGNED (31 downto 0)
    ) return UNSIGNED is
    begin
        if order_qty <= best_qty then
            return order_qty;
        else
            return best_qty;
        end if;
    end function;

begin
    -- Main matching process
    process(clk)
        variable match_possible : BOOLEAN;
        variable qty_to_match : UNSIGNED (31 downto 0);
    begin
        if rising_edge(clk) then
            if reset = '1' then
                match_state <= IDLE;
                match_occurred <= '0';
                match_price <= (others => '0');
                match_qty <= (others => '0');
                aggressor_id <= (others => '0');
                passive_id <= (others => '0');
                update_bid <= '0';
                update_ask <= '0';
                new_best_bid <= (others => '0');
                new_best_bid_qty <= (others => '0');
                new_best_ask <= (others => '0');
                new_best_ask_qty <= (others => '0');
                remaining_qty <= (others => '0');
                match_price_reg <= (others => '0');
                aggressor_id_reg <= (others => '0');
                passive_id_reg <= (others => '0');
            else
                case match_state is
                    when IDLE =>
                        match_occurred <= '0';
                        update_bid <= '0';
                        update_ask <= '0';

                        if new_order = '1' then
                            -- Check if order can match immediately
                            if order_side = '0' then  -- Buy order
                                match_possible := can_match('0', order_price, best_ask_price, best_ask_qty);
                                if match_possible then
                                    match_state <= CHECK_MATCH;
                                    match_price_reg <= best_ask_price;
                                    aggressor_id_reg <= order_id;
                                    -- Passive order ID would come from book state
                                    passive_id_reg <= x"000000000001";  -- Placeholder
                                else
                                    -- No match, add to bid book
                                    update_bid <= '1';
                                    if order_price > best_bid_price or best_bid_qty = 0 then
                                        new_best_bid <= order_price;
                                        new_best_bid_qty <= order_qty;
                                    else
                                        new_best_bid <= best_bid_price;
                                        new_best_bid_qty <= best_bid_qty;
                                    end if;
                                    match_state <= IDLE;
                                end if;
                            else  -- Sell order
                                match_possible := can_match('1', order_price, best_bid_price, best_bid_qty);
                                if match_possible then
                                    match_state <= CHECK_MATCH;
                                    match_price_reg <= best_bid_price;
                                    aggressor_id_reg <= order_id;
                                    passive_id_reg <= x"000000000002";  -- Placeholder
                                else
                                    -- No match, add to ask book
                                    update_ask <= '1';
                                    if order_price < best_ask_price or best_ask_qty = 0 then
                                        new_best_ask <= order_price;
                                        new_best_ask_qty <= order_qty;
                                    else
                                        new_best_ask <= best_ask_price;
                                        new_best_ask_qty <= best_ask_qty;
                                    end if;
                                    match_state <= IDLE;
                                end if;
                            end if;
                        end if;

                    when CHECK_MATCH =>
                        -- Calculate match quantity
                        if order_side = '0' then  -- Buy order
                            qty_to_match := calculate_match_qty(order_qty, best_ask_qty);
                        else  -- Sell order
                            qty_to_match := calculate_match_qty(order_qty, best_bid_qty);
                        end if;

                        remaining_qty <= order_qty - qty_to_match;
                        match_state <= EXECUTE_MATCH;

                    when EXECUTE_MATCH =>
                        -- Generate match output
                        match_occurred <= '1';
                        match_price <= match_price_reg;
                        match_qty <= order_qty - remaining_qty;
                        aggressor_id <= aggressor_id_reg;
                        passive_id <= passive_id_reg;

                        -- Update book based on remaining quantity
                        if remaining_qty > 0 then
                            -- Partial fill, add remaining to opposite book
                            if order_side = '0' then  -- Buy order
                                update_bid <= '1';
                                if order_price > best_bid_price or best_bid_qty = 0 then
                                    new_best_bid <= order_price;
                                    new_best_bid_qty <= remaining_qty;
                                else
                                    new_best_bid <= best_bid_price;
                                    new_best_bid_qty <= best_bid_qty;
                                end if;
                            else  -- Sell order
                                update_ask <= '1';
                                if order_price < best_ask_price or best_ask_qty = 0 then
                                    new_best_ask <= order_price;
                                    new_best_ask_qty <= remaining_qty;
                                else
                                    new_best_ask <= best_ask_price;
                                    new_best_ask_qty <= best_ask_qty;
                                end if;
                            end if;
                        else
                            -- Full fill, just update the matched side
                            if order_side = '0' then  -- Buy order filled ask
                                update_ask <= '1';
                                -- New best ask would come from book state
                                new_best_ask <= x"00000000";  -- Placeholder
                                new_best_ask_qty <= x"00000000";
                            else  -- Sell order filled bid
                                update_bid <= '1';
                                new_best_bid <= x"00000000";  -- Placeholder
                                new_best_bid_qty <= x"00000000";
                            end if;
                        end if;

                        match_state <= UPDATE_BOOK;

                    when UPDATE_BOOK =>
                        -- Hold book update for one cycle
                        update_bid <= '0';
                        update_ask <= '0';
                        match_state <= IDLE;

                    when others =>
                        match_state <= IDLE;
                end case;
            end if;
        end if;
    end process;

end Behavioral;
