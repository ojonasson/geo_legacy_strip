#!/usr/bin/env python3

import os
import sys
import netCDF4
import argparse
import re
import tempfile
import shutil
import numpy as np

legacy_regex_str =\
    r'ACSPO_([\w\.]+)_(\w+)_(\w+)_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})-(\d{2})(\d{2})_(\d{8}).(\d{6}).nc'

legacy_regex = re.compile(legacy_regex_str)

# precision (number of correct digits for each output layer).
default_precision_dict = {
    'latitude': 4,
    'longitude': 4,
    'satellite_zenith_angle': 3,
    'solar_zenith_angle': 2,
    'relative_azimuth_angle': 2,
    'solar_azimuth_angle': 2,
    'brightness_temp_ch7': 3,
    'brightness_temp_ch11': 3,
    'brightness_temp_ch13': 3,
    'brightness_temp_ch14': 3,
    'brightness_temp_ch15': 3,
    'brightness_temp_crtm_ch7': 3,
    'brightness_temp_crtm_ch11': 3,
    'brightness_temp_crtm_ch13': 3,
    'brightness_temp_crtm_ch14': 3,
    'brightness_temp_crtm_ch15': 3,
    'brightness_temp_crtm_ch7_sst': 3,
    'brightness_temp_crtm_ch11_sst': 3,
    'brightness_temp_crtm_ch13_sst': 3,
    'brightness_temp_crtm_ch14_sst': 3,
    'brightness_temp_crtm_ch15_sst': 3,
    'sst_regression': 3,
    'sens_regression': 3,
    'sses_bias_acspo': 3,
    'sst_reynolds': 3,
    'air_temp_gfs': 2,
    'u_wind_gfs': 2,
    'v_wind_gfs': 2,
    'tpw_acspo': 3,
}

default_hourly_layers = ('pixel_line_number',
                         'pixel_line_time',
                         'ascending_descending_flag',
                         'latitude',
                         'longitude',
                         'satellite_zenith_angle',
                         'solar_zenith_angle',
                         'relative_azimuth_angle',
                         'solar_azimuth_angle',
                         'brightness_temp_ch7',
                         'brightness_temp_ch11',
                         'brightness_temp_ch13',
                         'brightness_temp_ch14',
                         'brightness_temp_ch15',
                         'brightness_temp_crtm_ch7',
                         'brightness_temp_crtm_ch11',
                         'brightness_temp_crtm_ch13',
                         'brightness_temp_crtm_ch14',
                         'brightness_temp_crtm_ch15',
                         'brightness_temp_crtm_ch7_sst',
                         'brightness_temp_crtm_ch11_sst',
                         'brightness_temp_crtm_ch13_sst',
                         'brightness_temp_crtm_ch14_sst',
                         'brightness_temp_crtm_ch15_sst',
                         'acspo_mask',
                         'individual_clear_sky_tests_results',
                         'extra_byte_clear_sky_tests_results',
                         'sst_regression',
                         'sens_regression',
                         'sses_bias_acspo',
                         'sst_reynolds',
                         'air_temp_gfs',
                         'u_wind_gfs',
                         'v_wind_gfs',
                         'tpw_acspo')

default_non_hourly_layers = ('pixel_line_number',
                             'pixel_line_time',
                             'ascending_descending_flag',
                             # 'latitude',
                             # 'longitude',
                             # 'brightness_temp_ch7',
                             'brightness_temp_ch11',
                             'brightness_temp_ch13',
                             'brightness_temp_ch14',
                             'brightness_temp_ch15',
                             'acspo_mask',
                             'individual_clear_sky_tests_results',
                             'extra_byte_clear_sky_tests_results',
                             'sst_regression')


'''
Strip uneccessary (for collation) layers from legacy files and reduce precision of remaining layers
to a reasonable accuracy
path: path to geo legacy file
complevel: gzip (deflate) compression level. Must be within [0, 9]
output_dir: Output directory. It not supplied, input will be modified in place (overwritten)
'''


def strip_geo_legacy_file(path,
                          complevel=5,
                          output_dir=None,
                          precision_dict=None,
                          hourly_layers=None,
                          non_hourly_layers=None):

    if precision_dict is None:
        precision_dict = default_precision_dict

    if hourly_layers is None:
        hourly_layers = default_hourly_layers

    if non_hourly_layers is None:
        non_hourly_layers = default_non_hourly_layers

    name = os.path.basename(path)

    if not os.path.isfile(path):
        raise IOError('File "{}" does not exist')

    match = legacy_regex.match(name)

    if not match:
        raise IOError('File "{}" is not a legacy file')

    if complevel < 0 or complevel > 9:
        raise ValueError('Invalid commplevel={}. Must be within [0, 9]'.format(complevel))

    is_hourly = int(match[8]) == 0

    if is_hourly:
        output_layers = hourly_layers
    else:
        output_layers = non_hourly_layers

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, name)
        with netCDF4.Dataset(tmp_path, 'w') as ncf_tmp, netCDF4.Dataset(path, 'r') as ncf_in:

            dims = ncf_in.dimensions

            for dim_name, dim in dims.items():
                ncf_tmp.createDimension(dim.name, size=dim.size)

            for att_name in ncf_in.ncattrs():
                ncf_tmp.setncattr(att_name, ncf_in.getncattr(att_name))

            for varname in output_layers:

                try:
                    in_var = ncf_in.variables[varname]
                except Exception as e:
                    raise IOError('File "{}" is missing layer "{}": {}'.format(name, varname, str(e)))
                least_significant_digit = precision_dict.get(varname, None)

                out_var = ncf_tmp.createVariable(varname,
                                                 in_var.datatype,
                                                 in_var.dimensions,
                                                 zlib=True,
                                                 complevel=complevel,
                                                 shuffle=True,
                                                 least_significant_digit=least_significant_digit)

                for att_name in in_var.ncattrs():
                    out_var.setncattr(att_name, in_var.getncattr(att_name))

                data = in_var[:]
                if np.ma.isMaskedArray(data):
                    if data.dtype == np.float32:
                        data = data.filled(fill_value=np.nan)
                    else:
                        data = data.filled()

                out_var[:] = data

        if output_dir:
            out_path = os.path.join(output_dir, name)
        else:
            out_path = path

        shutil.copy2(tmp_path, out_path)

    return


def main():

    parser = argparse.ArgumentParser(
        description='Strip uneccessary (for collation) layers from legacy files and reduce '
                    'precision of remaining layers to a reasonable accuracy'
    )
    parser.add_argument('files', metavar='N', type=str, nargs='+', help='Space separated list of input files')
    parser.add_argument('--output_dir',
                        default='',
                        type=str,
                        help='Output directory. Iinput files will be overwritten ifno output directory is given')
    parser.add_argument('--compress_level', default=5, type=int, help='Output compression level')

    args = parser.parse_args()

    files = args.files
    output_dir = args.output_dir
    compress_level = args.compress_level

    for path in files:
        print('Stripping file "{}"'.format(os.path.basename(path)))
        strip_geo_legacy_file(path, complevel=compress_level, output_dir=output_dir)

    return 0


if __name__ == '__main__':

    sys.exit(main())
