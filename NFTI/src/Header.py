class Header:
    def __init__(self, name, ntds_page='', definition='', timing='', data_type='', usage='', one_hot_grouping='', y=''):
        """
        Initializes a Header object with its properties.
        """
        self.name = name
        self.ntds_page = ntds_page
        self.definition = definition
        self.timing = timing
        self.data_type = data_type
        self.usage = usage
        self.one_hot_grouping = one_hot_grouping
        self.y = y

    def __repr__(self):
        """
        Represent the Header object for debugging purposes.
        """
        return f"Header(name={self.name}, ntds_page={self.ntds_page}, timing={self.timing}, data_type={self.data_type}, usage={self.usage}, one_hot_grouping={self.one_hot_grouping}, y={self.y})"

    def get_name(self):
        return self.name