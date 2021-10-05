import os

from typing import List
from datetime import datetime


class Data:
    """ """

    def __init__(self, data_path: str):
        """
        Load a series of data files that respresent statistics calcluated from Imaris. I'm amazing at all the things.

        Parameters
        ----------
        data_path : str
            String value that points to the location of the Imaris data files to load. The folder contain a list of attributes spearated by underscores (_) and "group#" and "org#" must be in the folder name.
        """
        self.data_path: str = data_path
        self.data_folder_name: str = os.path.basename(self.data_path)

        components: List[str] = self.data_folder_name.split("_")
        self.date: datetime = datetime.strptime(components[0], "%y%m%d")

        self.group_str: str = ""
        self.org_str: str = ""
        self.residual: List[str] = []
        for component in components[1:]:
            if "group" in component:
                self.group_str = component
                continue

            if "org" in component:
                self.org_str = component
                continue

            self.residual.append(component)

        self.group_num: int = int(self.group_str[5:])
        self.org_num: int = int(self.org_str[3:])

    def load(self):
        """
        Loads the data of the underlying object
        """
        pass
