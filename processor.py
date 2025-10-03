import numpy as np
import h5netcdf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path

class MERRA2_Processor:
    """
    MERRA-2 Processor for the M2T1NXSLV and M2T1NXFLX collections with multi-file processing
    """
    
    def __init__(self, data_directory: str, slv_variables: list[str], flx_variables: list[str]):
        if not (slv_variables or flx_variables):
            raise Exception("Collection variables not provided")

        self.data_directory = Path(data_directory)
        self.fill_values = {}
        self.processed_files = set()
        
        # Define expected variables for each collection
        self.slv_variables = slv_variables
        self.flx_variables = flx_variables
        self.all_variables = self.slv_variables + self.flx_variables
        
        print(f"MERRA-2 Processor initialized")
        print(f"Data directory: {self.data_directory}")
        print(f"Target SLV variables: {self.slv_variables}")
        print(f"Target FLX variables: {self.flx_variables}")
    
    def find_merra2_files(self, date_range: tuple[datetime.date, datetime.date]=None):
        """
        Find all MERRA-2 SLV and FLX files in the directory
        Returns dictionary organized by date and collection type
        """
        file_patterns = {
            'SLV': '*tavg1_2d_slv_Nx*.nc',  # T1NXSLV files
            'FLX': '*tavg1_2d_flx_Nx*.nc'   # T1NXFLX files
        }
        
        found_files = {'SLV': {}, 'FLX': {}}
        
        for collection, pattern in file_patterns.items():
            files = list(self.data_directory.glob(pattern))
            
            print(f"🔍 Found {len(files)} {collection} files")
            
            for file_path in files:
                # Extract date from filename
                # Expected format: MERRA2_XXX.tavg1_2d_slv_Nx.YYYYMMDD.nc
                filename = file_path.name
                try:
                    # Find date pattern YYYYMMDD
                    import re
                    date_match = re.search(r'(\d{8})', filename)
                    if date_match:
                        date_str = date_match.group(1)
                        file_date = datetime.strptime(date_str, '%Y%m%d').date()
                        
                        # Filter by date range if specified
                        if date_range is None or (date_range[0] <= file_date <= date_range[1]):
                            found_files[collection][file_date] = file_path
                    else:
                        print(f"[WARNING] Could not extract date from filename: {filename}")
                        
                except Exception as e:
                    print(f"[ERROR] Error processing filename {filename}: {e}")
        
        # Show summary
        slv_dates = set(found_files['SLV'].keys())
        flx_dates = set(found_files['FLX'].keys())
        common_dates = slv_dates.intersection(flx_dates)
        
        print(f"\n[DEBUG] File Summary:")
        print(f"   SLV files: {len(slv_dates)} dates")
        print(f"   FLX files: {len(flx_dates)} dates")
        print(f"   Common dates (can be merged): {len(common_dates)} dates")
        
        if len(common_dates) < len(slv_dates.union(flx_dates)):
            missing_slv = flx_dates - slv_dates
            missing_flx = slv_dates - flx_dates
            if missing_slv:
                print(f"   [WARNING]  Missing SLV files for: {sorted(list(missing_slv))[:5]}...")
            if missing_flx:
                print(f"   [WARNING]  Missing FLX files for: {sorted(list(missing_flx))[:5]}...")
        
        return found_files, common_dates


def main():    
    processor = MERRA2_Processor(
        data_directory="./large",
        slv_variables=['QV2M', 'QV10M', 'U2M', 'U10M', 'V2M', 'V10M', 'T2M'],
        flx_variables=['PRECSNO', 'PRECTOT']
    )
    
    # Process files for a specific date range
    from datetime import date
    start_date = date(2023, 1, 1)
    end_date = date(2024, 12, 31)

    # test find files
    processor.find_merra2_files(date_range=(start_date, end_date))


if __name__ == "__main__":
    print("🌍 MERRA-2 Data Processor")
    print("This processor handles T1NXSLV and T1NXFLX collections\n")

    main()