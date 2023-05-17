class SearchConfig(object):

    def __init__(self):
        self.k = 4
        self.alpha = 0.6
        self.beam = 3
        self.maxSeqLength = 30
        self.eosTokenId = 50256
        self.padTokenId = 50256
        self.suffixPadding = True
        self.listSize = 15
