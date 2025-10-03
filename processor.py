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
        
        print(f"🌍 MERRA-2 Processor initialized")
        print(f"📁 Data directory: {self.data_directory}")
        print(f"🎯 Target SLV variables: {self.slv_variables}")
        print(f"🎯 Target FLX variables: {self.flx_variables}")
    
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
                        print(f"[WARNING]  Could not extract date from filename: {filename}")
                        
                except Exception as e:
                    print(f"[ERROR]  Error processing filename {filename}: {e}")
        
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
    
    def extract_fill_values(self, file_path, variables: list[str]):
        """Extract fill values from a single netCDF file"""
        try:
            with h5netcdf.File(file_path, mode="r") as f:
                fill_values = {}
                
                for var_name in variables:
                    if var_name not in f.variables:
                        continue
                        
                    var = f.variables[var_name]
                    fill_value = None
                    
                    # Check for fill value attributes
                    if hasattr(var, 'attrs'):
                        attrs = var.attrs
                        for attr_name in ['_FillValue', 'missing_value', 'fill_value']:
                            if attr_name in attrs:
                                fill_value = attrs[attr_name]
                                break
                    
                    # Auto-detect if not found
                    if fill_value is None:
                        try:
                            data_sample = var[0] if len(var.shape) > 2 else var[:]
                            data_flat = data_sample.flatten()
                            unique_vals = np.unique(data_flat)
                            large_vals = unique_vals[unique_vals > 1e10]
                            if len(large_vals) > 0:
                                fill_value = large_vals[0]
                        except:
                            fill_value = 9.999999999e+14  # Default
                    
                    fill_values[var_name] = fill_value
                
                return fill_values
                
        except Exception as e:
            print(f"⚠️  Error extracting fill values from {file_path}: {e}")
            return {}
    
    def process_single_file(self, file_path, collection_type: str, expected_variables):
        """
        Process a single MERRA-2 file (SLV or FLX)
        Returns cleaned data as pandas DataFrame
        """
        try:
            with h5netcdf.File(file_path, mode="r") as f:
                print(f"[PROCESSING] {collection_type} file: {file_path.name}")
                
                # Get coordinates
                if 'time' in f.variables:
                    times = f['time'][:]
                else:
                    # Some files might not have time dimension for daily averages
                    times = [0]  # Placeholder
                
                lats = f['lat'][:]
                lons = f['lon'][:]
                
                print(f"   Dimensions: {len(times)} times × {len(lats)} lats × {len(lons)} lons")
                print(f"   Lat range: {lats.min():.2f} to {lats.max():.2f}")
                print(f"   Lon range: {lons.min():.2f} to {lons.max():.2f}")
                
                # Extract date from filename for timestamp
                filename = file_path.name
                import re
                date_match = re.search(r'(\d{8})', filename)
                if date_match:
                    file_date = datetime.strptime(date_match.group(1), '%Y%m%d')
                else:
                    file_date = datetime.now()  # Fallback
                
                # Process each variable
                data_records = []
                
                for var_name in expected_variables:
                    if var_name not in f.variables:
                        print(f"   [WARNING]  Variable {var_name} not found in {collection_type} file")
                        continue
                    
                    var_data = f[var_name][:]
                    fill_value = self.fill_values.get(var_name, 9.999999999e+14)
                    
                    print(f"   📊 Processing {var_name}: shape {var_data.shape}")
                    
                    # Handle different data shapes
                    if len(var_data.shape) == 3:  # (time, lat, lon)
                        for t_idx in range(var_data.shape[0]):
                            timestamp = file_date + timedelta(hours=t_idx)  # Assuming hourly
                            
                            for lat_idx, lat in enumerate(lats):
                                for lon_idx, lon in enumerate(lons):
                                    value = var_data[t_idx, lat_idx, lon_idx]
                                    
                                    # Skip fill values
                                    if abs(value - fill_value) < 1e-6 or np.isnan(value):
                                        continue
                                    
                                    data_records.append({
                                        'timestamp': timestamp.timestamp(),
                                        'latitude': float(lat),
                                        'longitude': float(lon),
                                        'variable': var_name,
                                        'value': float(value),
                                        'collection': collection_type
                                    })
                    
                    elif len(var_data.shape) == 2:  # (lat, lon) - daily average
                        timestamp = file_date
                        
                        for lat_idx, lat in enumerate(lats):
                            for lon_idx, lon in enumerate(lons):
                                value = var_data[lat_idx, lon_idx]
                                
                                # Skip fill values
                                if abs(value - fill_value) < 1e-6 or np.isnan(value):
                                    continue
                                
                                data_records.append({
                                    'timestamp': timestamp.timestamp(),
                                    'latitude': float(lat),
                                    'longitude': float(lon),
                                    'variable': var_name,
                                    'value': float(value),
                                    'collection': collection_type
                                })
                    
                    else:
                        print(f"   [ERROR]  Unexpected shape for {var_name}: {var_data.shape}")
                        continue
                
                df = pd.DataFrame(data_records)
                print(f"   [SUCCESS] Extracted {len(df):,} valid data points from {collection_type}")
                
                return df
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return pd.DataFrame()
    
    def merge_slv_flx_data(self, slv_df, flx_df):
        """
        Merge SLV and FLX data for the same date/location
        """

        # base cases, empty-empty, empty-exists & exists-empty
        if slv_df.empty and flx_df.empty:
            return pd.DataFrame()
        elif slv_df.empty:
            return flx_df
        elif flx_df.empty:
            return slv_df
        
        # Combine dataframes
        combined_df = pd.concat([slv_df, flx_df], ignore_index=True)
        
        # Pivot to get variables as columns
        merged_df = combined_df.pivot_table(
            index=['timestamp', 'latitude', 'longitude'],
            columns='variable',
            values='value',
            aggfunc='first'  # Take first value if we encounter duplicates
        ).reset_index()

        # Flatten column names
        merged_df.columns.name = None
        
        print(merged_df)

        return merged_df
    
    def create_database(self, db_path):
        """Create database schema for MERRA-2 surface data"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Drop existing table if it exists
        cursor.execute('DROP TABLE IF EXISTS weather_data')
        
        # Create new table with all surface variables
        cursor.execute('''
        CREATE TABLE weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            latitude REAL,
            longitude REAL,
            
            -- Temperature and humidity (from SLV)
            T2M_Celsius REAL,      -- 2-meter air temperature (C), derived from T2M (Kelvin)
            QV2M REAL,             -- 2-meter specific humidity (kg/kg)
            QV10M REAL,            -- 10-meter specific humidity (kg/kg)
            
            -- Precipitation (from FLX)
            PRECSNO REAL,          -- Snowfall (kg/m^2/s)
            PRECTOT REAL,          -- Total precipitation (kg/m^2/s)
            
            -- Wind (derived from SLV's U2M, V2M, U10M and V10M)
            wind_speed_2m REAL,     -- 2-meter wind speed magnitude (m/s)
            wind_direction_2m REAL, -- 2-meter wind direction (degrees, 0=North, 90=East)

            wind_speed_10m REAL,      -- 10-meter wind speed magnitude (m/s)
            wind_direction_10m REAL,  -- 10-meter wind direction (degrees, 0=North, 90=East)
            
            UNIQUE(timestamp, latitude, longitude)
        )
        ''')
        
        conn.commit()
        conn.close()
        print(f"[SUCCESS] Database created: {db_path}")
    
    def process_multiple_files(self, db_path, date_range=None, batch_size=10):
        """
        Process multiple MERRA-2 files and store in database
        """
        # Find all files
        found_files, common_dates = self.find_merra2_files(date_range)
        
        if not common_dates:
            print("[WARNING] No matching SLV/FLX file pairs found!")
            return
        
        # Extract fill values from first files
        print("\nExtracting fill values...")
        first_slv = list(found_files['SLV'].values())[0]
        first_flx = list(found_files['FLX'].values())[0]
        
        slv_fills = self.extract_fill_values(first_slv, self.slv_variables)
        flx_fills = self.extract_fill_values(first_flx, self.flx_variables)
        
        self.fill_values.update(slv_fills)
        self.fill_values.update(flx_fills)
        
        print("Fill values detected:")
        for var, fill_val in self.fill_values.items():
            print(f"   {var}: {fill_val}")
        
        # Create database
        self.create_database(db_path)
        
        # Process files in batches
        sorted_dates = sorted(common_dates)
        total_dates = len(sorted_dates)
        
        print(f"\n[PROCESSING] {total_dates} date pairs in batches of {batch_size}")
        
        for i in range(0, total_dates, batch_size):
            batch_dates = sorted_dates[i:i+batch_size]
            batch_data = []
            
            print(f"\nProcessing batch {(i // batch_size ) + 1}/{(total_dates + batch_size - 1) // batch_size}")
            print(f"   Dates: {batch_dates[0]} to {batch_dates[-1]}")
            
            for date in batch_dates:
                try:
                    # Process SLV and FLX files for this date
                    slv_file = found_files['SLV'][date]
                    flx_file = found_files['FLX'][date]
                    
                    print(f"   Processing {date}...")
                    
                    # Process each file
                    slv_data = self.process_single_file(slv_file, 'SLV', self.slv_variables)
                    flx_data = self.process_single_file(flx_file, 'FLX', self.flx_variables)
                    
                    # Merge data
                    if not slv_data.empty or not flx_data.empty:
                        merged_data = self.merge_slv_flx_data(slv_data, flx_data)
                        
                        if not merged_data.empty:
                            # Add derived fields
                            merged_data = self.add_derived_fields(merged_data)
                            batch_data.append(merged_data)
                    
                except Exception as e:
                    print(f"   [ERROR]  Error processing {date}: {e}")
                    continue
            
            # Store batch in database
            if batch_data:
                combined_batch = pd.concat(batch_data, ignore_index=True)
                self.store_batch_in_database(combined_batch, db_path)
                print(f"   [SUCCESS]  Stored {len(combined_batch):,} records from batch")
            
        print(f"\n🎉 Processing complete! Data stored in {db_path}")
        self.show_database_summary(db_path)
    
    def add_derived_fields(self, df: pd.DataFrame):
        """Add computed fields to the merged data"""
        df = df.copy()
        
        # Convert temperature from Kelvin to Celsius
        if 'T2M' in df.columns:
            df['T2M_Celsius'] = df['T2M'] - 273.15

            # Drop redundant column
            df = df[df.columns.difference(["T2M"])]
        
        # Calculate wind speed magnitudes and directions (2m)
        if 'U2M' in df.columns and 'V2M' in df.columns:
            df['wind_speed_2m'] = np.sqrt(df['U2M']*df['U2M'] + df['V2M']*df['V2M'])
            
            # Wind direction in degrees (0deg = North, 90deg = East, etc.)
            df['wind_direction_2m'] = (90.0 - np.degrees(np.arctan2(df['V2M'], df['U2M']))) % 360.0

            # Drop redundant columns
            df = df[df.columns.difference(["U2M", "V2M"])]
        
        # Calculate wind speed magnitudes and directions (10m)
        if 'U10M' in df.columns and 'V10M' in df.columns:
            df['wind_speed_10m'] = np.sqrt(df['U10M']*df['U10M'] + df['V10M']*df['V10M'])
            
            # Wind direction in degrees (0deg = North, 90deg = East, etc.)
            df['wind_direction_10m'] = (90.0 - np.degrees(np.arctan2(df['V10M'], df['U10M']))) % 360.0

            # Drop redundant columns
            df = df[df.columns.difference(["U10M", "V10M"])]
        
        return df

    def store_batch_in_database(self, df: pd.DataFrame, db_path):
        """Store a batch of processed data in the database"""
        conn = sqlite3.connect(db_path)

        # converting PRECTOT and PRECSNO's unit from mm/s to mm/day
        df['PRECTOT'] = df['PRECTOT'] * 86400
        df['PRECSNO'] = df['PRECSNO'] * 86400

        # normalize the timestamp of each day
        df['timestamp'] = df['timestamp'].apply(lambda t: datetime.fromtimestamp(t).date())

        # take the daily mean for every distinct (time, lat, long) triple,
        # reduces the size of the data significantly
        df = df.groupby(['timestamp', 'latitude', 'longitude']).mean().reset_index()

        try:
            df.to_sql('weather_data', conn, if_exists='append', index=False, method='multi', chunksize=50)  # Faster bulk insert
        except Exception as e:
            print(f"[ERROR]  Database insert error: {e}")
            # Try row by row as fallback
            for _, row in df.iterrows():
                try:
                    row.to_frame().T.to_sql('weather_data', conn, if_exists='append', index=False)
                except Exception as e:
                    raise e
                    ## continue  # Skip problematic rows
        finally:
            conn.close()
    
    def show_database_summary(self, db_path):
        """Show summary statistics of the processed database"""
        conn = sqlite3.connect(db_path)
        
        # Basic statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM weather_data")
        time_range = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(DISTINCT latitude || ',' || longitude) FROM weather_data")
        unique_locations = cursor.fetchone()[0]
        
        print(f"\n[DEBUG] Database Summary:")
        print(f"   Total records: {total_records:,}")
        print(f"   Time range: {time_range[0]} to {time_range[1]}")
        print(f"   Unique locations: {unique_locations:,}")
        
        # Variable coverage
        print(f"\n[DEBUG] Variable Coverage:")
        for var in (set(self.all_variables) - {'U2M', 'U10M', 'V2M', 'V10M', 'T2M'}).union({'T2M_Celsius', 'wind_speed_2m', 'wind_speed_10m', 'wind_direction_2m', 'wind_direction_10m'}):
            cursor.execute(f"SELECT COUNT(*) FROM weather_data WHERE {var} IS NOT NULL")
            count = cursor.fetchone()[0]
            coverage = ((count / total_records) * 100) if total_records > 0 else 0
            print(f"   {var}: {count:,} records ({coverage:.1f}%)")
        
        conn.close()

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
    
    # Process and store in database
    processor.process_multiple_files(
        db_path="large_merra2_data.db",
        date_range=(start_date, end_date),
        batch_size=1  # Process 10 days at a time
    )

if __name__ == "__main__":
    print("MERRA-2 Data Processor")
    print("This processor handles T1NXSLV and T1NXFLX collections\n")

    main()