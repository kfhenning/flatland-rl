class TestDef:
    def __init__(self, test_id, n_agents, x_dim, y_dim, n_cities
                 #,n_rails_in_min,n_rails_in_max,n_rails_between_min,n_rails_between_max
                 , n_runs):
        self.test_id = test_id
        self.n_agents = n_agents
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_cities = n_cities
        # self.rails_in_min=n_rails_in_min
        # self.rails_in_max = n_rails_in_max
        # self.n_rails_between_min = n_rails_between_min
        # self.n_rails_between_max = n_rails_between_max
        self.n_runs = n_runs



class EnvConfig:

    def __init__(self):
        tests = []
        tests.append(TestDef("Test_00", 5, 25, 25, 2, 50))
        tests.append(TestDef("Test_01", 10, 30, 30, 2, 50))
        tests.append(TestDef("Test_02", 20, 30, 30, 3, 50))
        tests.append(TestDef("Test_03", 50, 20, 35, 3, 40))
        tests.append(TestDef("Test_04", 80, 35, 20, 5, 30))
        tests.append(TestDef("Test_05", 80, 35, 35, 5, 30))
        tests.append(TestDef("Test_06", 80, 40, 60, 9, 30))
        tests.append(TestDef("Test_07", 80, 60, 40, 13, 30))
        tests.append(TestDef("Test_08", 80, 60, 60, 17, 20))
        tests.append(TestDef("Test_09", 100, 80, 120, 21, 20))
        tests.append(TestDef("Test_10", 100, 100, 80, 25, 20))
        tests.append(TestDef("Test_11", 200, 100, 100, 29, 10))
        tests.append(TestDef("Test_12", 200, 150, 150, 33, 10))
        tests.append(TestDef("Test_13", 400, 150, 150, 37, 10))
        tests.append(TestDef("Test_14", 425, 158, 158, 41, 10))

        self.tests = tests



if __name__ == "__main__":
    print("run")
    tests = EnvConfig().tests
    print(tests[7].test_id)
