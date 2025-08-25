import torch
from torch.utils._sympy.functions import ModularIndexing, FloorDiv
import sympy
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class Term:
    coefficient: int
    range: Optional[int]  # None for unbounded
    original_expr: sympy.Expr
    reconstruction_multiplier: int  # The multiplier needed for reconstruction

def analyze_invertible_indexing(expr: sympy.Expr, var: sympy.Symbol) -> Optional[Tuple[List[sympy.Expr], sympy.Expr]]:
    """
    Analyze an expression to see if it's an invertible indexing pattern.
    
    Returns:
        None if not invertible, or
        (extraction_exprs, reconstruction_expr) where:
        - extraction_exprs: [expr0, expr1, ...] for extracting components from y
        - reconstruction_expr: full expression to reconstruct var from the components
    """
    
    # Step 1: Parse all terms
    terms = parse_terms(expr, var)
    if not terms:
        return None
    
    # Step 2: Sort by coefficient (descending)
    terms.sort(key=lambda t: t.coefficient, reverse=True)
    
    # Step 3: Check invertibility conditions
    if not check_invertibility(terms):
        return None
    
    # Step 4: Generate extraction and reconstruction
    extraction_exprs = generate_extraction_exprs(terms)
    reconstruction_expr = generate_reconstruction_expr(terms, var)
    
    return extraction_exprs, reconstruction_expr

def parse_terms(expr: sympy.Expr, var: sympy.Symbol) -> List[Term]:
    """Parse expression into terms."""
    if not isinstance(expr, sympy.Add):
        # Single term
        term = parse_single_term(expr, var)
        return [term] if term else []
    
    terms = []
    for arg in expr.args:
        term = parse_single_term(arg, var)
        if term:
            terms.append(term)
        else:
            return []  # If any term fails to parse, fail completely
    
    return terms

def parse_single_term(term: sympy.Expr, var: sympy.Symbol) -> Optional[Term]:
    """Parse a single term and extract coefficient, range, and reconstruction multiplier."""
    
    # Extract coefficient
    coefficient = 1
    expr = term
    
    if isinstance(term, sympy.Mul):
        const_parts = []
        non_const_parts = []
        
        for arg in term.args:
            if arg.is_number:
                const_parts.append(arg)
            else:
                non_const_parts.append(arg)
        
        if const_parts:
            coefficient = int(sympy.Mul(*const_parts))
            if len(non_const_parts) == 1:
                expr = non_const_parts[0]
            elif len(non_const_parts) == 0:
                # Constant term
                return Term(
                    coefficient=coefficient, 
                    range=1, 
                    original_expr=sympy.S.One,
                    reconstruction_multiplier=0
                )
            else:
                return None
    
    # Now determine the range and reconstruction multiplier
    range_val, reconstruction_multiplier = analyze_expression_properties(expr, var)
    if reconstruction_multiplier is None:
        return None
        
    return Term(
        coefficient=coefficient, 
        range=range_val, 
        original_expr=expr,
        reconstruction_multiplier=reconstruction_multiplier
    )

def analyze_expression_properties(expr: sympy.Expr, var: sympy.Symbol) -> Tuple[Optional[int], Optional[int]]:
    """Analyze an expression to determine its range and reconstruction multiplier."""
    
    # ModularIndexing(var, divisor, modulo) = (var // divisor) % modulo
    if isinstance(expr, ModularIndexing):
        x, div, mod = expr.args
        if x == var:
            div_val = int(div)
            mod_val = int(mod)
            return mod_val, div_val  # Range is mod, multiplier is div
    
    # FloorDiv cases
    if isinstance(expr, FloorDiv):
        base, divisor = expr.args
        div_val = int(divisor)
        
        # FloorDiv(ModularIndexing(var, 1, mod), div) = (var % mod) // div
        if isinstance(base, ModularIndexing):
            x, inner_div, mod = base.args
            if x == var and int(inner_div) == 1:
                mod_val = int(mod)
                range_val = mod_val // div_val
                return range_val, div_val  # Range is mod//div, multiplier is div
        
        # FloorDiv(var, divisor) = var // divisor (unbounded)
        elif base == var:
            return None, div_val  # Unbounded range, multiplier is div
    
    # Just the variable itself
    if expr == var:
        return None, 1  # Unbounded range, multiplier is 1
    
    return None, None

def check_invertibility(terms: List[Term]) -> bool:
    """Check if the terms represent an invertible transformation."""
    
    # Coefficients must be strictly decreasing
    coeffs = [t.coefficient for t in terms]
    if coeffs != sorted(coeffs, reverse=True):
        return False
    
    # For invertibility, each coefficient should equal the product of ranges of subsequent terms
    bounded_terms = [t for t in terms if t.range is not None]
    if len(bounded_terms) < len(terms):
        # For now, assume invertible if we have an unbounded leading term
        return True
    
    # Check that coefficients match range products
    for i in range(len(bounded_terms) - 1):
        expected_coef = 1
        for j in range(i + 1, len(bounded_terms)):
            expected_coef *= bounded_terms[j].range
        
        if bounded_terms[i].coefficient != expected_coef:
            return False
    
    return True

def generate_extraction_exprs(terms: List[Term]) -> List[sympy.Expr]:
    """Generate the full extraction expressions from y."""
    y = sympy.Symbol('y')
    exprs = []
    remainder = y
    
    for i, term in enumerate(terms):
        if i < len(terms) - 1:
            # Extract this component
            component_expr = remainder // term.coefficient
            exprs.append(component_expr)
            remainder = remainder % term.coefficient
        else:
            # Last term gets the remainder
            exprs.append(remainder)
    
    return exprs

def generate_reconstruction_expr(terms: List[Term], var: sympy.Symbol) -> sympy.Expr:
    """Generate the full reconstruction expression."""
    # y = sympy.Symbol('y')
    y = var
    reconstruction = sympy.S.Zero
    remainder = y
    
    for i, term in enumerate(terms):
        if i < len(terms) - 1:
            component = FloorDiv(remainder, term.coefficient)
            remainder = ModularIndexing(remainder, 1, term.coefficient)
        else:
            component = remainder
        
        # Use the pre-computed reconstruction multiplier
        reconstruction += component * term.reconstruction_multiplier
    
    return reconstruction

# def create_forward_function(original_expr: sympy.Expr, var: sympy.Symbol):
#     """Create a function that evaluates the original expression."""
#     def forward(value: int) -> int:
#         return int(original_expr.subs(var, value))
#     return forward

# def test_transformation():
#     """Test the transformation with the example from the prompt."""
#     from sympy import Symbol
    
#     p0 = Symbol('p0')
    
#     expr = (
#         16384 * FloorDiv(p0, 16384) +
#         4096 * FloorDiv(ModularIndexing(p0, 1, 16), 4) +
#         128 * ModularIndexing(p0, 16, 32) +
#         4 * ModularIndexing(p0, 512, 32) +
#         ModularIndexing(p0, 1, 4)
#     )
    
#     result = analyze_invertible_indexing(expr, p0)
#     if not result:
#         print("❌ Transformation not recognized as invertible")
#         return False
    
#     extraction_exprs, reconstruction_expr = result
#     print("✅ Transformation analyzed successfully")
#     print("\nExtraction expressions:")
#     for i, expr_item in enumerate(extraction_exprs):
#         print(f"  d{i} = {expr_item}")
#     print(f"\nReconstruction: p0 = {reconstruction_expr}")
    
#     # Create forward function
#     forward_fn = create_forward_function(expr, p0)
    
#     return test_inverse_correctness(forward_fn, expr, p0, extraction_exprs, reconstruction_expr)

# def test_inverse_correctness(forward_fn, original_expr, var, extraction_exprs, reconstruction_expr):
#     """Test that the inverse formula is correct"""
#     print("\n" + "="*50)
#     print("TESTING INVERSE CORRECTNESS")
#     print("="*50)
    
#     # Test with comprehensive set of values
#     test_values = [
#         0, 1, 2, 3, 4, 15, 16, 31, 32, 63, 64, 127, 128,
#         511, 512, 1023, 1024, 4095, 4096, 16383, 16384, 16385,
#         32767, 32768, 65535, 65536, 131071
#     ]
    
#     # Add some random values
#     import random
#     random.seed(42)
#     for _ in range(50):
#         test_values.append(random.randint(0, 1000000))
    
#     errors = 0
#     y_symbol = sympy.Symbol('y')
    
#     for p0_val in test_values:
#         # Compute forward transformation
#         y_val = forward_fn(p0_val)
        
#         # Compute inverse using symbolic expressions
#         p0_recovered = int(reconstruction_expr.subs(y_symbol, y_val))
        
#         if p0_val != p0_recovered:
#             errors += 1
#             print(f"❌ ERROR at p0={p0_val}:")
#             print(f"   Forward: y = {y_val}")
#             print(f"   Inverse: p0_recovered = {p0_recovered}")
            
#             # Debug: show the extracted components
#             print("   Extraction steps:")
#             remainder = y_val
#             for i, expr_item in enumerate(extraction_exprs):
#                 component_val = int(expr_item.subs(y_symbol, y_val))
#                 print(f"     d{i} = {component_val}")
            
#             if errors >= 5:  # Limit error output
#                 break
    
#     if errors == 0:
#         print(f"✅ All {len(test_values)} test cases passed!")
#         test_bijection_property(forward_fn, min(100000, max(test_values)))
#         return True
#     else:
#         print(f"❌ {errors} errors out of {len(test_values)} test cases")
#         return False
# # 
# def test_bijection_property(forward_fn, max_test_val):
#     """Test that the formula is a bijection (one-to-one)"""
#     print(f"\n🔍 Testing bijection property for values 0 to {max_test_val}...")
    
#     outputs = set()
#     for p0 in range(min(max_test_val, 10000)):  # Limit for performance
#         y = forward_fn(p0)
#         if y in outputs:
#             print(f"❌ Not injective! Duplicate output {y} for different inputs")
#             return False
#         outputs.add(y)
    
#     print(f"✅ Tested {min(max_test_val, 10000)} values, all unique outputs (bijective)")
#     return True

# def analyze_bit_patterns():
#     """Analyze the bit extraction patterns"""
#     print("\n" + "="*50)
#     print("BIT EXTRACTION ANALYSIS")
#     print("="*50)
    
#     terms_info = [
#         ("16384 * (p0 // 16384)", "Extracts bits [14, ∞)", "Places at bit 14+"),
#         ("4096 * ((p0 % 16) // 4)", "Extracts bits [2, 4) from p0 % 16", "Places at bit 12+"),
#         ("128 * ((p0 // 16) % 32)", "Extracts bits [4, 9)", "Places at bit 7+"),
#         ("4 * ((p0 // 512) % 32)", "Extracts bits [9, 14)", "Places at bit 2+"),
#         ("(p0 % 4)", "Extracts bits [0, 2)", "Places at bit 0")
#     ]
    
#     for term, extraction, placement in terms_info:
#         print(f"{term:25} → {extraction:30} → {placement}")

# if __name__ == "__main__":
#     success = test_transformation()
#     if success:
#         analyze_bit_patterns()
#     else:
#         print("\n❌ Tests failed - transformation may be incorrect")