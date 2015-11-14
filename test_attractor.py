from attractor import Attractor

def test_self.dt():
    """Testing that the initialization parameters are being calculated correctly."""
    dt_true = (80.0 - 0.0)/10000
    print("Expected dt:", dt_true)
    a = Attractor()
    print("Actual dt:", a.dt)
    assert a.dt == dt_true

def test_deriv():
    """Testing the derivative calculated.

    This is an important test because the euler, rk2, and rk4 estimates use this method.  If this method does not calculate the correct values, then             everything else will be incorrect.
    """
    deriv_true = [10.0,  23.0,  -6.0]
    print("Expected deriv:", deriv_true)
    a = Attractor()
    print("Actual deriv:", a.deriv([1,2,3]))
    assert a.deriv([1,2,3]) == deriv_true

def test_euler():
    """Testing the euler method.

    This is being tested because this is a main method in this class.
    """
    euler_true = [0.08, 0.184, -0.048]
    print("Expected euler:", euler_true)
    a = Attractor()
    print("Actual euler:", a.euler([1,2,3]))
    assert a.euler([1,2,3]) == euler_true

def test_rk2():
    """Testing the Runge-Kutta method of order 2.

    This is being tested because it uses a new increment which increases the chance of error.
    """
    rk2_true = [0.08416, 0.19146368, -0.04608256]
    print("Expected rk2:", rk2_true)
    a = Attractor()
    print("Actual rk2:", a.rk2([1,2,3]))
    assert a.rk2([1,2,3]) == rk2_true

def test_rk4():
    """Testing the Runge-Kutta method of order 4.

    This is being tested because it has the most amount steps to calculate the result.  This is the most likely place for errors.
    """
    rk4_true = [0.08425138,  0.19172248, -0.04604073]
    print("Expected rk4:", r4_true)
    a = Attractor()
    print("Actual rk4:", a.rk4([1,2,3]))
    assert a.rk4([1,2,3]) == rk4_true