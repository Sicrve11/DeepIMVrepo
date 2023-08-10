import os, shutil, glob
from selenium import webdriver
import time

# tumor_list = [
#     'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD',
#     'DLBC', 'ESCA', 'FPPP', 'GBM', 'GBMLGG', 'HNSC', 'KICH',
#     'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD',
#     'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
#     'SARC', 'SKCM', 'STAD', 'STES', 'TGCT', 'THCA',
#     'THYM', 'UCEC', 'UCS', 'UVM'
# ]
# 'FPPP' 'LAML'
tumor_list = [
    'LGG', 'LIHC', 'LUAD',
    'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
    'SARC', 'SKCM', 'STAD', 'STES', 'TGCT', 'THCA',
    'THYM', 'UCEC', 'UCS', 'UVM'
]

filelist = [
    "Methylation_Preprocess.Level_3.2016012800.0.0.tar.gz",
    "miRseq_Preprocess.Level_3.2016012800.0.0.tar.gz",
    "mRNAseq_Preprocess.Level_3.2016012800.0.0.tar.gz",
    "RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz",
    "Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
]

def downloadTCGAdata(dlp, filepath):
    for name in tumor_list:
        url = f"http://firebrowse.org/?cohort={name}&download_dialog=true"

        # 打开网页
        driver = webdriver.Chrome(executable_path=r"E:\ChromeDriver\chromedriver.exe")
        driver.get(url)
        time.sleep(5)

        for filetype in filelist:
            driver.find_element_by_link_text(filetype.split('.')[0]).click()

        # 检测是否下载完成
        while True:
            downlist = glob.glob(f"{dlp}/*.tar.gz")
            if len(downlist) == len(filelist):
                print(f"{name} 下载完成！")
                break
            else:
                time.sleep(1)

        # 移动文件
        filepath = rootpath + name
        if not os.path.exists(filepath):
            os.mkdir(filepath)
            for filetype in filelist:
                downpath = f"{dlp}/gdac.broadinstitute.org_{name}.{filetype}"
                filen = f"{filepath}/gdac.broadinstitute.org_{name}.{filetype}"
                shutil.move(downpath, filen)

        driver.quit()       # 关闭浏览器
        print(f"{name} 完成！")


dlp = "C:/Users/PC/Downloads"
rootpath = "E:/CellTech/ST_Tools/VAE_data/dataset/TCGA/"

downloadTCGAdata(dlp, rootpath)

