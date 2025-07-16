from mpmath import zeta, mp

# Set precision (you can increase this if needed)
mp.dps = 50  # 50 decimal places

def is_non_trivial_zero(s, tol=1e-10):
    """
    Check if s is a non-trivial zero of the Riemann zeta function.
    
    Args:
        s: Complex number
        tol: Tolerance for considering the result zero
    
    Returns:
        bool
    """
    if not (0 < s.real < 1):
        return False  # Not in the range of non-trivial zeros
    
    val = zeta(s)
    return abs(val) < tol

if __name__ == "__main__":
    # Example usage
    test_zero = 0.5 + 14.134725j  # First non-trivial zero on the critical line
    print(is_non_trivial_zero(test_zero))  # Should return True