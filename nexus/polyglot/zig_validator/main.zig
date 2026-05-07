const std = @import("std");

const Order = struct {
    symbol: []const u8,
    qty: i32,
    side: []const u8,
    price: f64,
};

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    if (args.len < 2) {
        try stdout.print("{{\"valid\": false, \"error\": \"No input\"}}\n", .{});
        return;
    }

    var parser = std.json.Parser.init(std.heap.page_allocator, false);
    defer parser.deinit();

    var tree = try parser.parse(args[1]);
    defer tree.deinit();

    // Low-level validation logic
    const root = tree.root.Object;
    const qty = root.get("qty").?.Integer;
    const price = root.get("price").?.Float;

    if (qty <= 0) {
        try stdout.print("{{\"valid\": false, \"error\": \"Quantity must be positive\"}}\n", .{});
        return;
    }

    if (price <= 0.0) {
        try stdout.print("{{\"valid\": false, \"error\": \"Price must be positive\"}}\n", .{});
        return;
    }

    try stdout.print("{{\"valid\": true, \"error\": null}}\n", .{});
}
