//===- QuantOps.td - Quantitative Finance Dialect Operations ---*- tablegen -*-===//
//
// Nexus Core Custom Compiler Infrastructure
//
// We define a domain-specific intermediate representation (IR) for quantitative 
// finance. Standard compilers like GCC/Clang do not understand financial mathematics.
// By defining "EMA" (Exponential Moving Average) or "BlackScholes" as a fundamental 
// compiler operation, we can optimize the execution graph *before* it hits x86_64,
// achieving mathematically-provable optimal latency.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Quant_Dialect : Dialect {
  let name = "quant";
  let summary = "A dialect for high-frequency quantitative trading operations.";
  let description = [{
    The `quant` dialect provides high-level mathematical abstractions for finance,
    allowing the MLIR optimizer to fuse stochastic equations, eliminate dead 
    trades, and vectorize order-book parsing using AVX-512 prior to LLVM lowering.
  }];
  let cppNamespace = "::nexus::compiler::quant";
}

// Base class for Quant operations
class Quant_Op<string mnemonic, list<Trait> traits = []> :
    Op<Quant_Dialect, mnemonic, traits>;

// An operation representing an Exponential Moving Average (EMA)
def Quant_EMAOp : Quant_Op<"ema", [Pure]> {
  let summary = "Exponential Moving Average of a time-series tensor";
  
  let arguments = (ins 
    F64Tensor:$prices,
    F64Attr:$alpha
  );
  
  let results = (outs F64Tensor:$result);

  let description = [{
    Calculates the EMA natively. The compiler knows the mathematical 
    properties of EMA and can fuse it with other momentum indicators 
    to prevent multiple passes over the L1 CPU cache.
  }];
}

// An operation representing hardware-accelerated Black-Scholes
def Quant_BlackScholesOp : Quant_Op<"black_scholes", [Pure]> {
  let summary = "European option pricing formula";
  
  let arguments = (ins 
    F64:$spot,
    F64:$strike,
    F64:$time_to_expiry,
    F64:$volatility,
    F64:$risk_free_rate
  );
  
  let results = (outs F64:$call_price, F64:$put_price);
}
