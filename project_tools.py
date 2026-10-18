import numpy as np
def load_data_csv(filepath: str):
    """Read and clean raw meteorological CSV file from exel.
    Replaces comma decimal separators with dots, removes empty
    trailing semicolons, empty bottom rows, and returns as numpy array.

    Keyword arguments:
    filepath -- relative path to the raw csv file

    Returns:
    cleanned_data -- numpy array of shape with float values
    """
    rows = []
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(';')[:14]
        if not parts[0] or not parts[0].isdigit():
            continue
        numeric_row = [float(val.replace(',', '.')) for val in parts]
        rows.append(numeric_row)
    return np.array(rows)



def calculate_cold_season_precip(precip_matrix: np.ndarray):
    """Calculate total snow/winter precipitation (Nov-Dec of prev year +Jan-Mar).

    Keyword arguments:
    precip_matrix -- 2D numpy array of monthly precipitation (N, 14)

    Returns:
    cold_precip -- 1D numpy array of cold season precipitation sums (mm)
    """
    jan_feb_mar = precip_matrix[:, 1] + precip_matrix[:, 2] + precip_matrix[:, 3]
    cold_precip = np.zeros(len(precip_matrix))
    cold_precip[0] = jan_feb_mar[0] + precip_matrix[0, 11] + precip_matrix[0, 12]
    cold_precip[1:] = (precip_matrix[:-1, 11] + precip_matrix[:-1, 12] + jan_feb_mar[1:])
    return cold_precip




def filter_flood_risk_years(years: np.ndarray, cold_precip: np.ndarray, apr_temp: np.ndarray):
    """Return boolean mask for years with high snow accumulation and warm April.

    Keyword arguments:
    years -- numpy array of years
    cold_precip -- numpy array of cold season precipitation (mm)
    apr_temp -- numpy array of April mean temperatures (C)

    Returns:
    risk_mask -- boolean numpy array (True for high flood risk years)
    """
    precip_norm = np.mean(cold_precip)
    temp_norm = np.mean(apr_temp)
    risk_mask = (cold_precip > precip_norm) & (apr_temp > temp_norm)
    return risk_mask
