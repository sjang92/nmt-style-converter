class AutomatorError(Exception):
    """Generic errors raised by automators"""
    def __init__(self, driver, msg=None):
        if msg is None:
            #Default useful error message
            msg = "Automator error"
        super(AutomatorError, self).__init__(msg)
        self.car = car

class AutomatorSetupError(Exception):
    """Setup error for automators"""
    def __init__(self, driver, user, pw):
        pass
