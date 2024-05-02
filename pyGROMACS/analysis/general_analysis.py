import numpy as np
import pandas as pd

def read_gromacs_xvg(file_path: str, fraction: float=0.0 ):
    """
    Reads data from a Gromacs XVG file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the XVG file.
    - fraction (float, optional): The fraction of data to select. Defaults to 0.0.

    Returns:
    - pandas.DataFrame: A DataFrame containing the selected data.

    Description:
    This function reads data from a Gromacs XVG file specified by 'file_path'. It extracts the data columns and their corresponding properties from the file. The data is then filtered based on the 'fraction' parameter, selecting only the data points that are within the specified fraction of the maximum time value. The selected data is returned as a pandas DataFrame, where each column represents a property and each row represents a data point.

    Example:
    read_gromacs_xvg('data.xvg', fraction=0.5)
    """
    data               = []
    properties         = []
    units              = []
    special_properties = []
    special_means      = []
    special_stds       = []
    special_units      = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('@') and "title" in line:
                title = line.split('"')[1]
                continue
            if line.startswith("@") and "xaxis  label" in line:
                properties.append( line.split('"')[1].split()[0] )
                units.append( line.split('"')[1].split()[1])
                continue
            if line.startswith("@") and "yaxis  label" in line:
                for u in line.split('"')[1].split(","):
                    # Check if this is a special file with several properties
                    if len(u.split("(")) > 1 and u.split("(")[0].strip():
                        properties.append( u.split()[0] )
                        units.append( u.split()[1].replace(r"\\N","").replace(r"\\S","^") )
                    else:
                        units.append( "(" + u.split("(")[1].replace(")","").replace(" ",".") + ")" )
                continue
            if line.startswith('@') and ("s" in line and "legend" in line):
                if "=" in line:
                    special_properties.append( line.split('"')[1].split("=")[0].replace(" ", "") )
                    mean, std, unit = [ a.replace(")","").replace(" ", "").replace("+/-","") for a in line.split('"')[1].split("=")[1].split("(") ]
                    special_means.append(mean)
                    special_stds.append(std)
                    special_units.append(f"({unit})")
                else:
                    properties.append( line.split('"')[1] )
                continue
            elif line.startswith('@') or line.startswith('#'):
                continue  # Skip comments and metadata lines
            parts = line.split()
            data.append([float(part) for part in parts])  

    # Create column wise array with data
    data = np.array([np.array(column) for column in zip(*data)])

    # In case special properties are delivered, there is just one regular property, which is given for every special property.
    if special_properties:
        properties[-1:] = [ properties[-1] + "[" + re.search(r'\[(.*?)\]', sp).group(1) + "]" for sp in special_properties ]
        units[-1:]      = [ units[-1] for _ in special_properties ]
    
    # Only select data that is within (fraction,1)*t_max
    idx = data[0] > fraction * data[0][-1]

    # Save data
    property_dict = {}

    for p, d, u in zip( properties, data[:,idx], units ):
        property_dict[f"{p} {u}"] = d

    # As special properties only have a mean and a std. Generate a series that exactly has the mean and the standard deviation of this value.
    for sp, sm, ss, su in zip( special_properties, special_means, special_stds, special_units ):
        property_dict[f"{sp} {su}"] =  generate_series( desired_mean = float(sm), desired_std = float(ss), size = len(data[0,idx]))

    return pd.DataFrame(property_dict)