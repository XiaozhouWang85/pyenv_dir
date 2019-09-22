import ftplib
import os
import tarfile

def ftp_download(url,output_filename=None):
	print("Downloading from {}".format(url))
	ftp_path = url.split("://")[-1]
	ftp_split = ftp_path.split("/")
	dir_path = "/".join(ftp_split[1:len(ftp_split)-1])
	ftp_site = ftp_split[0]
	ftp_site_filename = ftp_split[-1]
	if output_filename==None:
		output_filename=ftp_site_filename
	
	ftp = ftplib.FTP(ftp_site) 
	ftp.login() 
	ftp.cwd(dir_path)
	ftp.retrbinary("RETR " + ftp_site_filename, open(output_filename, 'wb').write)
	ftp.quit()

def tar_extract(fname):
	tar = tarfile.open(fname, "r:gz")
	tar.extractall()
	tar.close()

'''
directory = "downloads"
if not os.path.exists(directory):
    os.makedirs(directory)

ftp_download("ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz","downloads/siftsmall.tar.gz")
ftp_download("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz","downloads/sift.tar.gz")
'''
tar_extract("downloads/siftsmall.tar.gz")
tar_extract("downloads/sift.tar.gz")