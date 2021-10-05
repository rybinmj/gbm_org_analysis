from data import Data

class Organioid(Data):

    def __init__(self, data_path: str):
        """
        """
        super().__init__(data_path)
        assert "vol" in self.residual
