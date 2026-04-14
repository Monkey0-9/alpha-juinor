library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity pcie_dma is
    Port ( clk : in STD_LOGIC;
           reset : in STD_LOGIC;
           pcie_data_in : in STD_LOGIC_VECTOR (255 downto 0);
           dma_write_en : in STD_LOGIC;
           dma_addr : in UNSIGNED (31 downto 0);
           host_mem_data : out STD_LOGIC_VECTOR (255 downto 0));
end pcie_dma;

architecture Behavioral of pcie_dma is
    -- Simplified DMA Controller for PCIe Gen4 x16
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                host_mem_data <= (others => '0');
            elsif dma_write_en = '1' then
                -- Direct Memory Access logic to stream book updates to Host CPU
                host_mem_data <= pcie_data_in;
            end if;
        end if;
    end process;
end Behavioral;
