library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.MATH_REAL.ALL;

entity order_book is
    Port (
        clk : in STD_LOGIC;
        reset : in STD_LOGIC;
        -- Order input: [127:96]=price, [95:64]=quantity, [63:56]=side, [55:48]=symbol, [47:0]=order_id
        order_in : in STD_LOGIC_VECTOR (127 downto 0);
        order_valid : in STD_LOGIC;
        -- Best bid/ask outputs: [63:32]=price, [31:0]=quantity
        best_bid : out STD_LOGIC_VECTOR (63 downto 0);
        best_ask : out STD_LOGIC_VECTOR (63 downto 0);
        -- Matched trades
        trade_out : out STD_LOGIC_VECTOR (127 downto 0);
        trade_valid : out STD_LOGIC;
        -- Status
        book_full : out STD_LOGIC
    );
end order_book;

architecture Behavioral of order_book is
    -- Price level structure
    type price_level_type is record
        price : unsigned(31 downto 0);
        total_quantity : unsigned(31 downto 0);
        head_order_id : unsigned(47 downto 0);
        next_level_ptr : integer range 0 to 1023;
        prev_level_ptr : integer range 0 to 1023;
    end record;

    -- Order structure
    type order_type is record
        order_id : unsigned(47 downto 0);
        quantity : unsigned(31 downto 0);
        price : unsigned(31 downto 0);
        side : STD_LOGIC;  -- '0' = bid, '1' = ask
        next_order_ptr : integer range 0 to 2047;
        level_ptr : integer range 0 to 1023;
    end record;

    -- Arrays for price levels and orders
    type bid_levels_array is array (0 to 511) of price_level_type;
    type ask_levels_array is array (0 to 511) of price_level_type;
    type orders_array is array (0 to 2047) of order_type;

    -- Book storage
    signal bid_levels : bid_levels_array;
    signal ask_levels : ask_levels_array;
    signal orders : orders_array;

    -- Pointers for free lists
    signal free_level_ptr : integer range 0 to 511 := 0;
    signal free_order_ptr : integer range 0 to 2047 := 0;

    -- Best bid/ask level pointers
    signal best_bid_level : integer range 0 to 511 := 0;
    signal best_ask_level : integer range 0 to 511 := 0;

    -- Order parsing
    signal order_price : unsigned(31 downto 0);
    signal order_quantity : unsigned(31 downto 0);
    signal order_side : STD_LOGIC;
    signal order_symbol : unsigned(7 downto 0);
    signal order_id : unsigned(47 downto 0);

    -- Trade assembly
    signal trade_price : unsigned(31 downto 0);
    signal trade_quantity : unsigned(31 downto 0);
    signal trade_buyer_id : unsigned(47 downto 0);
    signal trade_seller_id : unsigned(47 downto 0);

    -- Helper functions
    function find_or_create_level(
        price : unsigned(31 downto 0);
        side : STD_LOGIC;
        levels : bid_levels_array
    ) return integer is
        variable level_idx : integer := 0;
        variable found : boolean := false;
    begin
        -- Search for existing price level
        for i in 0 to 511 loop
            if levels(i).price = price and levels(i).total_quantity > 0 then
                level_idx := i;
                found := true;
                exit;
            end if;
        end loop;

        -- Create new level if not found
        if not found and free_level_ptr < 512 then
            level_idx := free_level_ptr;
            free_level_ptr <= free_level_ptr + 1;
        end if;

        return level_idx;
    end function;

    function match_order(
        order_price : unsigned(31 downto 0);
        order_quantity : unsigned(31 downto 0);
        order_side : STD_LOGIC
    ) return unsigned is
        variable match_qty : unsigned(31 downto 0) := (others => '0');
        variable remaining_qty : unsigned(31 downto 0) := order_quantity;
    begin
        if order_side = '0' then  -- Buy order, match against asks
            for i in 0 to 511 loop
                if ask_levels(i).price <= order_price and ask_levels(i).total_quantity > 0 then
                    if ask_levels(i).total_quantity >= remaining_qty then
                        match_qty := match_qty + remaining_qty;
                        remaining_qty := (others => '0');
                    else
                        match_qty := match_qty + ask_levels(i).total_quantity;
                        remaining_qty := remaining_qty - ask_levels(i).total_quantity;
                    end if;
                end if;
                exit when remaining_qty = 0;
            end loop;
        else  -- Sell order, match against bids
            for i in 0 to 511 loop
                if bid_levels(i).price >= order_price and bid_levels(i).total_quantity > 0 then
                    if bid_levels(i).total_quantity >= remaining_qty then
                        match_qty := match_qty + remaining_qty;
                        remaining_qty := (others => '0');
                    else
                        match_qty := match_qty + bid_levels(i).total_quantity;
                        remaining_qty := remaining_qty - bid_levels(i).total_quantity;
                    end if;
                end if;
                exit when remaining_qty = 0;
            end loop;
        end if;

        return match_qty;
    end function;

begin
    -- Parse incoming order
    order_price <= unsigned(order_in(127 downto 96));
    order_quantity <= unsigned(order_in(95 downto 64));
    order_side <= order_in(63);
    order_symbol <= unsigned(order_in(55 downto 48));
    order_id <= unsigned(order_in(47 downto 0));

    -- Main matching engine process
    process(clk)
        variable matched_qty : unsigned(31 downto 0);
        variable remaining_qty : unsigned(31 downto 0);
        variable level_idx : integer;
    begin
        if rising_edge(clk) then
            if reset = '1' then
                -- Reset all book structures
                bid_levels <= (others => (
                    price => (others => '0'),
                    total_quantity => (others => '0'),
                    head_order_id => (others => '0'),
                    next_level_ptr => 0,
                    prev_level_ptr => 0
                ));
                ask_levels <= (others => (
                    price => (others => '0'),
                    total_quantity => (others => '0'),
                    head_order_id => (others => '0'),
                    next_level_ptr => 0,
                    prev_level_ptr => 0
                ));
                orders <= (others => (
                    order_id => (others => '0'),
                    quantity => (others => '0'),
                    price => (others => '0'),
                    side => '0',
                    next_order_ptr => 0,
                    level_ptr => 0
                ));
                free_level_ptr <= 0;
                free_order_ptr <= 0;
                best_bid_level <= 0;
                best_ask_level <= 0;
                trade_valid <= '0';
                book_full <= '0';
            else
                trade_valid <= '0';

                if order_valid = '1' then
                    -- Check for immediate matches
                    matched_qty := match_order(order_price, order_quantity, order_side);
                    remaining_qty := order_quantity - matched_qty;

                    if matched_qty > 0 then
                        -- Generate trade output
                        trade_price <= order_price;
                        trade_quantity <= matched_qty;
                        trade_valid <= '1';
                    end if;

                    -- Add remaining quantity to book
                    if remaining_qty > 0 then
                        if order_side = '0' then  -- Buy order
                            level_idx := find_or_create_level(order_price, '0', bid_levels);
                            if level_idx < 512 then
                                bid_levels(level_idx).price <= order_price;
                                bid_levels(level_idx).total_quantity <=
                                    bid_levels(level_idx).total_quantity + remaining_qty;
                                bid_levels(level_idx).head_order_id <= order_id;
                            end if;
                        else  -- Sell order
                            level_idx := find_or_create_level(order_price, '1', ask_levels);
                            if level_idx < 512 then
                                ask_levels(level_idx).price <= order_price;
                                ask_levels(level_idx).total_quantity <=
                                    ask_levels(level_idx).total_quantity + remaining_qty;
                                ask_levels(level_idx).head_order_id <= order_id;
                            end if;
                        end if;
                    end if;

                    -- Update best bid/ask pointers
                    for i in 0 to 511 loop
                        if bid_levels(i).total_quantity > 0 then
                            best_bid_level <= i;
                            exit;
                        end if;
                    end loop;

                    for i in 0 to 511 loop
                        if ask_levels(i).total_quantity > 0 then
                            best_ask_level <= i;
                            exit;
                        end if;
                    end loop;

                    -- Check if book is full
                    if free_level_ptr >= 511 or free_order_ptr >= 2047 then
                        book_full <= '1';
                    else
                        book_full <= '0';
                    end if;
                end if;
            end if;
        end if;
    end process;

    -- Output best bid/ask
    best_bid <= std_logic_vector(
        bid_levels(best_bid_level).price &
        bid_levels(best_bid_level).total_quantity
    );
    best_ask <= std_logic_vector(
        ask_levels(best_ask_level).price &
        ask_levels(best_ask_level).total_quantity
    );

    -- Assemble trade output
    trade_out <= std_logic_vector(
        trade_price &
        trade_quantity &
        trade_buyer_id &
        trade_seller_id
    );

end Behavioral;
