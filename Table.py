class Table:
    def __init__(self, Name, Data):
        self.name = Name
        self.data = Data

    def get_name(self):
        return self.name

    def get_data(self):
        return self.data.copy()

    def get_data_copy(self):
        return self.data.copy(deep=True)

