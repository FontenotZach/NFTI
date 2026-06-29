class Header:
    def __init__(self, name, ntds_page='', definition='', timing='', data_type='', load='', usage='', y=''):
        """
        Initializes a Header object with its properties.

        ``load`` controls whether the header is loaded into memory (populated into
        TraumaRecord.data). ``usage`` controls whether the header is used as a model
        training feature (in combination with timing/type).
        """
        self.name = name
        self.ntds_page = ntds_page
        self.definition = definition
        self.timing = timing
        self.data_type = data_type
        self.load = load
        self.usage = usage
        self.y = y

    def __repr__(self):
        """
        Represent the Header object for debugging purposes.
        """
        return f"Header(name={self.name}, ntds_page={self.ntds_page}, timing={self.timing}, data_type={self.data_type}, load={self.load}, usage={self.usage}, y={self.y})"

    def get_name(self):
        return self.name
