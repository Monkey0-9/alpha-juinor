//! Extreme Low-Latency Level 3 (Market-by-Order) Book
//!
//! Firms like XTX Markets and HRT require microscopic visibility into queue 
//! position and individual order dynamics. Standard books allocate memory
//! when an order arrives, causing GC or allocator spikes.
//! 
//! This implementation uses a pre-allocated static Arena (Slab allocator) 
//! meaning ZERO heap allocations happen during live trading.

use std::collections::HashMap;

/// Pre-allocated capacity to prevent memory resizing during trading hours
const MAX_LIVE_ORDERS: usize = 1_000_000;

#[derive(Clone, Copy)]
pub struct OrderNode {
    pub order_id: u64,
    pub price: u64,
    pub quantity: u32,
    pub prev_idx: usize,
    pub next_idx: usize,
    pub is_active: bool,
}

pub struct ZeroAllocationL3Book {
    /// The memory arena. No `Box` or `malloc` is ever called after initialization.
    arena: Vec<OrderNode>,
    free_head: usize,
    
    /// Fast O(1) lookup from OrderID to Arena Index
    order_map: HashMap<u64, usize>,
    
    /// Bids and Asks pointing to head indices of price levels
    bids: HashMap<u64, usize>,
    asks: HashMap<u64, usize>,
}

impl ZeroAllocationL3Book {
    pub fn new() -> Self {
        let mut arena = vec![
            OrderNode {
                order_id: 0, price: 0, quantity: 0, 
                prev_idx: 0, next_idx: 0, is_active: false
            }; 
            MAX_LIVE_ORDERS
        ];

        // Initialize free list
        for i in 0..MAX_LIVE_ORDERS - 1 {
            arena[i].next_idx = i + 1;
        }

        Self {
            arena,
            free_head: 0,
            order_map: HashMap::with_capacity(MAX_LIVE_ORDERS),
            bids: HashMap::with_capacity(10000),
            asks: HashMap::with_capacity(10000),
        }
    }

    #[inline(always)]
    pub fn add_order(&mut self, order_id: u64, price: u64, quantity: u32, is_bid: bool) {
        // Fast O(1) allocation from pre-warmed Slab
        let new_idx = self.free_head;
        self.free_head = self.arena[new_idx].next_idx;

        let node = &mut self.arena[new_idx];
        node.order_id = order_id;
        node.price = price;
        node.quantity = quantity;
        node.is_active = true;

        // O(1) Map insertion
        self.order_map.insert(order_id, new_idx);

        // Intrusive linked-list insertion logic would follow here...
        // (Omitted standard pointer manipulation for brevity)
    }

    #[inline(always)]
    pub fn cancel_order(&mut self, order_id: u64) {
        if let Some(&idx) = self.order_map.get(&order_id) {
            // Unlink from price level
            // ...

            // Return to free list
            self.arena[idx].is_active = false;
            self.arena[idx].next_idx = self.free_head;
            self.free_head = idx;
            
            self.order_map.remove(&order_id);
        }
    }
}
