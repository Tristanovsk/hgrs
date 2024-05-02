import ftplib
import os, glob
import time

opj=os.path.join
hostname = "131.176.221.216"
username = "acixIII-aqua"
password = "mSvpRoFsUmE69r--"

odir = "/_RESULTS"

workdir_ = '/media/harmel/TOSHIBA EXT/acix-iii/v2'
sites = ['Wendtorf', 'Varese', 'Geneve', 'Venice_Lagoon', 'Garda', 'Trasimeno',
         'AERONET-OC/galataplatform', 'AERONET-OC/sanmarcoplatform', 'AERONET-OC/zeebrugge', 'AERONET-OC/lisco',
         'AERONET-OC/lakeerie', 'AERONET-OC/casablanca', 'AERONET-OC/irbelighthouse',
         'AERONET-OC/ariaketower', 'AERONET-OC/kemigawa', 'AERONET-OC/uscseaprism', 'AERONET-OC/section7',
         'AERONET-OC/southgreenbay', 'AERONET-OC/palgrunden', 'AERONET-OC/venezia', 'AERONET-OC/lucinda',
         'AERONET-OC/bahiablanca', 'AERONET-OC/socheongcho', 'AERONET-OC/wavecissite', 'AERONET-OC/lakeokeechobee', 'AERONET-OC/gustavdalentower'
         ]
sites = [ 'Garda', 'Trasimeno',
         'AERONET-OC/galataplatform', 'AERONET-OC/sanmarcoplatform', 'AERONET-OC/zeebrugge', 'AERONET-OC/lisco',
         'AERONET-OC/lakeerie', 'AERONET-OC/casablanca', 'AERONET-OC/irbelighthouse',
         'AERONET-OC/ariaketower', 'AERONET-OC/kemigawa', 'AERONET-OC/uscseaprism', 'AERONET-OC/section7',
         'AERONET-OC/southgreenbay', 'AERONET-OC/palgrunden', 'AERONET-OC/venezia', 'AERONET-OC/lucinda',
         'AERONET-OC/bahiablanca', 'AERONET-OC/socheongcho', 'AERONET-OC/wavecissite', 'AERONET-OC/lakeokeechobee', 'AERONET-OC/gustavdalentower'
         ]
# workdir_ = '/data/satellite/acix-iii/AERONET-OC'
# sites = []



for site in sites:
    site_=site.replace('AERONET-OC/','')
    local_dir=opj(workdir_,site_)
    # Storing files to upload
    file_list = glob.glob(opj(local_dir,'*hGRSv2*.nc'))
    remote_dir = opj(odir,site,'hgrs')
    # Opening and connecting
    with ftplib.FTP(host=hostname, user=username, passwd=password) as ftp:

        # Navigating to remote directory
        ftp.cwd(remote_dir)

        # Accessing each file from file_list
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            print(file_path)
            if os.path.isfile(file_path):

                # Uploading file
                with open(file_path, 'rb') as file:
                    ftp.storbinary(f'STOR {file_name}', file)

                # Removing file from local directory after success
                os.remove(file_path)
