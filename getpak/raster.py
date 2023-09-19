import os
import json
import fiona
import numpy as np
import rasterio
import rasterio.mask
import scipy.ndimage
import importlib_resources
import xarray as xr
from dask.distributed import Client

from datetime import datetime
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling

from getpak.commons import Utils as u

class Raster:
    """
    Generic class containing methods for matricial manipulations
    
    Methods
    -------
    array2tiff(ndarray_data, str_output_file, transform, projection, no_data=-1, compression='COMPRESS=PACKBITS')
        Given an input ndarray and the desired projection parameters, create a raster.tif using rasterio.

    array2tiff_gdal(ndarray_data, str_output_file, transform, projection, no_data=-1, compression='COMPRESS=PACKBITS')
        Given an input ndarray and the desired projection parameters, create a raster.tif using GDT_Float32.
    
    reproj(in_raster, out_raster, target_crs='EPSG:4326')
        Given an input raster.tif reproject it to reprojected.tif using @target_crs

    shp_stats(tif_file, shp_poly, keep_spatial=True, statistics='count min mean max median std')
        Given a single-band GeoTIFF file and a vector.shp return statistics inside the polygon.

    extract_px(rasterio_rast, shapefile, rrs_dict, bands)
        Given a dict of Rrs and a polygon, to extract the values of pixels from each band

    sam(self, values, single=False)
        Given a a set of pixels, uses the Spectral Angle Mapper to generate angle between the Rrs and the OWTs

    classify_owt_px(self, rrs_dict, bands)
        Function to classify the OWT of each pixel

    classify_owt(self, rasterio_rast, shapefiles, rrs_dict, bands, min_px=6)
        Function to classify the the OWT of pixels inside a shapefile (or a set of shapefiles)

    """
    def __init__(self, parent_log=None):
            if parent_log:
                self.log = parent_log
            else:
                INSTANCE_TIME_TAG = datetime.now().strftime('%Y%m%dT%H%M%S')
                logfile = os.path.join(os.getcwd(), 'getpak_raster_' + INSTANCE_TIME_TAG + '.log')
            # Import CRS projection information from /data/s2_proj_ref.json
            s2projdata = importlib_resources.files(__name__).joinpath('data/s2_proj_ref.json')
            with s2projdata.open('rb') as fp:
                byte_content = fp.read()
            self.s2projgrid = json.loads(byte_content)

            # Import OWT means for S2 MSI from /data/means_OWT_S2A_Spyrakos.json
            means_owt = importlib_resources.files(__name__).joinpath('data/means_OWT_S2A_Spyrakos.json')
            with means_owt.open('rb') as fp:
                byte_content = fp.read()
            self.owts = dict(json.loads(byte_content))

    @staticmethod
    def array2tiff(ndarray_data, str_output_file, transform, projection, no_data=-1, compression='COMPRESS=PACKBITS'):
        """
        Given an input ndarray and the desired projection parameters, create a raster.tif using GDT_Float32.
        
        Parameters
        ----------
        @param ndarray_data: Inform if the index should be saved as array in the output folder
        @param str_output_file: string of the path of the file to be written
        @param transform: rasterio affine transformation matrix (resolution and "upper left" coordinate)
        @param projection: projection CRS
        @param no_data: the value for no data
        @param compression: type of file compression

        @return: None (If all goes well, array2tiff should pass and generate a file inside @str_output_file)
        """
        with rasterio.open(fp=str_output_file,
                           mode='w',
                           driver='GTiff',
                           height=ndarray_data.shape[0],
                           width=ndarray_data.shape[1],
                           count=1,
                           dtype=ndarray_data.dtype,
                           crs=projection,
                           transform=transform,
                           nodata=no_data,
                           options=['COMPRESS=PACKBITS']) as dst:
            dst.write(ndarray_data, 1)

        pass

    @staticmethod
    def array2tiff_gdal(ndarray_data, str_output_file, transform, projection, no_data=-1, compression='COMPRESS=PACKBITS'):
        """
        Given an input ndarray and the desired projection parameters, create a raster.tif using GDT_Float32.

        Parameters
        ----------
        @param ndarray_data: Inform if the index should be saved as array in the output folder
        @param str_output_file:
        @param transform:
        @param projection: projection CRS
        @param no_data: the value for no data
        @param compression: type of file compression

        @return: None (If all goes well, array2tiff should pass and generate a file inside @str_output_file)
        """
        # Create file using information from the template
        outdriver = gdal.GetDriverByName("GTiff")  # http://www.gdal.org/gdal_8h.html
        # imgs_out = /work/scratch/guimard/grs2spm/
        [cols, rows] = ndarray_data.shape
        # GDT_Byte = 1, GDT_UInt16 = 2, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6,
        # options=['COMPRESS=PACKBITS'] -> https://gdal.org/drivers/raster/gtiff.html#creation-options
        outdata = outdriver.Create(str_output_file, rows, cols, 1, gdal.GDT_Float32, options=[compression])
        # Write the array to the file, which is the original array in this example
        outdata.GetRasterBand(1).WriteArray(ndarray_data)
        # Set a no data value if required
        outdata.GetRasterBand(1).SetNoDataValue(no_data)
        # Georeference the image
        outdata.SetGeoTransform(transform)
        # Write projection information
        outdata.SetProjection(projection)
        # Close the file https://gdal.org/tutorials/raster_api_tut.html#using-create
        outdata = None
        pass
    
    @staticmethod
    def reproj(in_raster, out_raster, target_crs='EPSG:4326'):
        """
        Given an input raster.tif reproject it to reprojected.tif using @target_crs (default = 'EPSG:4326').
        
        Parameters
        ----------
        @param in_raster: Inform if the index should be saved as array in the output folder
        @param out_raster:
        @param target_crs:
        
        @return: None (If all goes well, reproj should pass and generate a reprojected file inside @out_raster)
        """
        # Open the input raster file
        with rasterio.open(in_raster) as src:

            # Calculate the transformation parameters
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            # Define the output file metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Create the output raster file
            with rasterio.open(out_raster, 'w', **kwargs) as dst:

                # Reproject the data
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest)
        print(f'Done: {out_raster}')
        pass
    
    @staticmethod
    def shp_stats(tif_file, shp_poly, keep_spatial=False, statistics='count min mean max median std'):
        """
        Given a single-band GeoTIFF file and a vector.shp return statistics inside the polygon.
        
        Parameters
        ----------
        @param tif_file: path to raster.tif file.
        @param shp_poly: path to the polygon.shp file.
        @param keep_spatial (bool): 
            True = include the input shp_poly in the output as GeoJSON 
            False (default) = get only the mini_raster and statistics
        @param statistics: what to extract from the shapes, available values are:
        
        min, max, mean, count, sum, std, median, majority,
        minority, unique, range, nodata, percentile.
        https://pythonhosted.org/rasterstats/manual.html#zonal-statistics        
        
        @return: roi_stats (dict) containing the extracted statistics inside the region of interest.
        """
        # with fiona.open(shp_poly) as src:
        #     roi_stats = zonal_stats(src,
        #                             tif_file,
        #                             stats=statistics,
        #                             raster_out=True,
        #                             all_touched=True,
        #                             geojson_out=keep_spatial,
        #                             band=1)
        # # Original output comes inside a list containing only the output dict:
        # return roi_stats[0]
        roi_stats = zonal_stats(shp_poly,
                                tif_file,
                                stats=statistics,
                                raster_out=True,
                                all_touched=True,
                                geojson_out=keep_spatial,
                                band=1)
        # Original output comes inside a list containing only the output dict:
        return roi_stats[0]

    @staticmethod
    def extract_px(rasterio_rast, shapefile, rrs_dict, bands):
        """
        Given a dict of Rrs and a polygon, to extract the values of pixels from each band

        Parameters
        ----------
        rasterio_rast: a rasterio raster open with rasterio.open
        shapefile: a polygon opened as geometry using fiona
        rrs_dict: a dict containing the Rrs bands
        bands: an array containing the bands of the rrs_dict to be extracted

        Returns
        -------
        values: an array of dimension n, where n is the number of bands, containing the values inside the polygon
        slice: the slice of the polygon (from the rasterio window)
        mask_image: the rasterio mask
        """
        # rast = rasterio_rast.read(1)
        mask_image, _, window_image = rasterio.mask.raster_geometry_mask(rasterio_rast, [shapefile], crop=True)
        slices = window_image.toslices()
        values = []
        for band in bands:
            # subsetting the xarray dataset
            subset_data = rrs_dict[band].isel(x=slices[1], y=slices[0])
            # Extract values where mask_image is False
            values.append(subset_data.where(~mask_image).values.flatten())

        return values, slices, mask_image

    def sam(self, rrs, single=False):
        """
        Spectral Angle Mapper for OWT classification for a set of pixels
        It calculates the angle between the Rrs of a set of pixels and those of the 13 OWT of inland waters (Spyrakos et al., 2018)
        Input values are the values of the pixels from B2 to B7, the dict of the OWTs is already stored
        Returns the spectral angle between the Rrs of the pixels and each OWT
        To classify pixels individually, set single=True
        ----------
        """
        if single:
            E = rrs / rrs.sum()
        else:
            med = np.nanmean(rrs, axis=1)
            E = med / med.sum()
        # norm of the vector
        nE = np.linalg.norm(E)

        # Convert owts values to numpy array for vectorized computations
        M = np.array([list(val.values()) for val in self.owts.values()])
        nM = np.linalg.norm(M, axis=1)

        # scalar product
        num = np.dot(M, E)
        den = nM * nE

        angles = np.arccos(num / den) * 180 / np.pi

        return int(np.argmin(angles) + 1)

    def classify_owt_px(self, rrs_dict, bands=['Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7']):
        """
        Function to classify the OWT of each pixel

        Parameters
        ----------
        rrs_dict: a xarray Dataset containing the Rrs bands
        bands: an array containing the bands (2 to 7) of the rrs_dict to be extracted

        Returns
        -------
        class_px: an array, with the same size as the input bands, with the pixels classified
        """
        # Drop the variables that won't be used in the classification
        variables_to_drop = [var for var in rrs_dict.variables if var not in bands]
        rrs = rrs_dict.drop_vars(variables_to_drop)
        # Find non-NaN values
        nzero = np.where(~np.isnan(rrs[bands[0]].values))
        # array of OWT class for each pixel
        class_px = np.zeros_like(rrs[bands[0]], dtype='uint8')
        # array of angles to limit the loop
        angles = np.zeros((len(nzero[0])), dtype='uint8')
        # loop over each nonzero value in the rrs_dict
        pix = np.zeros((len(nzero[0]), len(bands)))
        for i in range(len(bands)):
            pix[:, i] = rrs[bands[i]].values[nzero]
        for i in range(len(nzero[0])):
            angles[i] = self.sam(rrs=pix[i, :], single=True)
        class_px[nzero] = angles

        return class_px

    def classify_owt(self, rasterio_rast, shapefiles, rrs_dict, bands=['Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6', 'Rrs_B7'], min_px=6):
        """
        Function to classify the the OWT of pixels inside a shapefile (or a set of shapefiles)

        Parameters
        ----------
        rasterio_rast: a rasterio raster open with rasterio.open
        shapefiles: a polygon (or set of polygons), usually of waterbodies to be classified, opened as geometry using fiona
        rrs_dict: a dict containing the Rrs bands
        bands: an array containing the bands of the rrs_dict to be extracted
        min_px: minimum number of pixels in each polygon to operate the classification

        Returns
        -------
        class_spt: an array, with the same size as the input bands, with the classified pixels
        class_shp: an array with the same length as the shapefiles, with a OWT class for each polygon
        """
        class_spt = np.zeros(rrs_dict[bands[0]].shape, dtype='uint8')
        class_shp = np.zeros((len(shapefiles)), dtype='uint8')
        for i, shape in enumerate(shapefiles):
            values, slices, mask = self.extract_px(rasterio_rast, shape, rrs_dict, bands)
            # Verifying if there are more pixels than the minimum
            valid_pixels = np.isnan(values[0]) == False
            if np.count_nonzero(valid_pixels) >= min_px:
                angle = self.sam(values)
            else:
                angle = int(0)

            # classifying only the valid pixels inside the polygon
            values = np.where(valid_pixels, angle, 0)
            class_spt[slices[0], slices[1]] = values.reshape((mask.shape))
            # classification by polygon
            class_shp[i] = angle

        return class_spt, class_shp

class GRS:
    """
    Core functionalities to handle GRS files

    Methods
    -------
    metadata(grs_file_entry)
        Given a GRS string element, return file metadata extracted from its name.
    """

    def __init__(self, parent_log=None):
            if parent_log:
                self.log = parent_log
    
    @staticmethod
    def metadata(grs_file_entry):
        """
        Given a GRS file return metadata extracted from its name:
        
        Parameters
        ----------
        @param grs_file_entry (str or pathlike obj): path that leads to the GRS.nc file.
                
        @return: metadata (dict) containing the extracted info, available keys are:
            input_file, basename, mission, prod_lvl, str_date, pydate, year,
            month, day, baseline_algo_version, relative_orbit, tile, 
            product_discriminator, cloud_cover, grs_ver.
        
        Reference
        ---------
        Given the following file:
        /root/23KMQ/2021/05/21/S2A_MSIL1C_20210521T131241_N0300_R138_T23KMQ_20210521T163353_cc020_v15.nc
        
        S2A : (MMM) is the mission ID(S2A/S2B)
        MSIL1C : (MSIXXX) Product procesing level
        20210521T131241 : (YYYYMMDDTHHMMSS) Sensing start time
        N0300 : (Nxxyy) Processing Baseline number
        R138 : Relative Orbit number (R001 - R143)
        T23KMQ : (Txxxxx) Tile Number
        20210521T163353 : Product Discriminator
        cc020 : GRS Cloud cover estimation (0-100%)
        v15 : GRS algorithm baseline version

        For GRS version >=2.0, the naming does not include the cloud cover estimation nor the GRS version

        Further reading:
        Sentinel-2 MSI naming convention:
        URL = https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention
        """
        metadata = {}
        basefile = os.path.basename(grs_file_entry)
        splt = basefile.split('_')
        if len(splt)==9:
            mission,proc_level,date_n_time,proc_ver,r_orbit,tile,prod_disc,cc,aux = basefile.split('_')
            ver,_ = aux.split('.')
        else:
            mission,proc_level,date_n_time,proc_ver,r_orbit,tile,aux = basefile.split('_')
            prod_disc,_ = aux.split('.')
            cc, ver = ['NA', 'v20']
        file_event_date = datetime.strptime(date_n_time, '%Y%m%dT%H%M%S')
        yyyy = f"{file_event_date.year:02d}"
        mm = f"{file_event_date.month:02d}"
        dd = f"{file_event_date.day:02d}"
        
        metadata['input_file'] = grs_file_entry
        metadata['basename'] = basefile
        metadata['mission'] = mission
        metadata['prod_lvl'] = proc_level
        metadata['str_date'] = date_n_time
        metadata['pydate'] = file_event_date
        metadata['year'] = yyyy
        metadata['month'] = mm
        metadata['day'] = dd
        metadata['baseline_algo_version'] = proc_ver
        metadata['relative_orbit'] = r_orbit
        metadata['tile'] = tile
        metadata['product_discriminator'] = prod_disc
        metadata['cloud_cover'] = cc
        metadata['grs_ver'] = ver

        return metadata
    
    @staticmethod
    def get_grs_dict(grs_nc_file, grs_version='v15'):
        '''
        Opens the GRS netCDF files using the xarray library and dask, returning a DataArray containing only the Rrs bands
        Parameters
        ----------
        grs_nc_file: the path to the GRS file
        grs_version: a string with GRS version ('v15' or 'v20' for version 2.0.5)

        Returns
        -------
        The xarray DataArray containing the 11 Rrs bands, named as 'Rrs_B*'
        '''
        # list of bands
        bands = ['Rrs_B1', 'Rrs_B2', 'Rrs_B3', 'Rrs_B4', 'Rrs_B5', 'Rrs_B6',
                 'Rrs_B7', 'Rrs_B8', 'Rrs_B8A', 'Rrs_B11', 'Rrs_B12']
        if grs_version == 'v15':
            ds = xr.open_dataset(grs_nc_file, engine="netcdf4", decode_coords='all', chunks={'y': 610, 'x': 610})
            # List of variables to keep
            if 'Rrs_B1' in ds.variables:
                variables_to_keep = bands
                # Drop the variables you don't want
                variables_to_drop = [var for var in ds.variables if var not in variables_to_keep]
                grs = ds.drop_vars(variables_to_drop)
        elif grs_version == 'v20':
            ds = xr.open_dataset(grs_nc_file, chunks={'y': 610, 'x': 610}, engine="netcdf4")
            waves = ds['Rrs']['wl']
            subset_dict = {band: ds['Rrs'].sel(wl=waves[i]).drop(['wl', 'x', 'y']) for i, band in enumerate(bands)}
            grs = xr.Dataset(subset_dict)
        else:
            grs = None
        ds.close()

        return grs
    
    def param2tiff(self, ndarray_data, img_ref, output_img, no_data=0, gdal_driver_name="GTiff"):
        
        # Gather information from the template file
        ref_data = gdal.Open(img_ref)
        trans = ref_data.GetGeoTransform()
        proj = ref_data.GetProjection()
        # nodatav = 0 #data.GetNoDataValue()
        # Create file using information from the template
        outdriver = gdal.GetDriverByName(gdal_driver_name)  # http://www.gdal.org/gdal_8h.html
        
        [cols, rows] = ndarray_data.shape
        
        print(f'Writing output .tiff')
        # GDT_Byte = 1, GDT_UInt16 = 2, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6,
        # options=['COMPRESS=PACKBITS'] -> https://gdal.org/drivers/raster/gtiff.html#creation-options
        outdata = outdriver.Create(output_img, rows, cols, 1, gdal.GDT_Float32, options=['COMPRESS=PACKBITS'])
        # Write the array to the file, which is the original array in this example
        outdata.GetRasterBand(1).WriteArray(ndarray_data)
        # Set a no data value if required
        outdata.GetRasterBand(1).SetNoDataValue(no_data)
        # Georeference the image
        outdata.SetGeoTransform(trans)
        # Write projection information
        outdata.SetProjection(proj)

        # Closing the files
        # https://gdal.org/tutorials/raster_api_tut.html#using-create
        data = None
        outdata = None
        self.log.info('')
        pass

#    def proc_grs_wd_intersect(self, grs_dict, ):
