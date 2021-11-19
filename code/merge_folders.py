import shutil, os

RootDir1 = "E:/ISY5003-IRS-Practice-Module/vision/captured_images/target_b_data"
TargetFolder = "E:/ISY5003-IRS-Practice-Module/vision/captured_images/target_b"
for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
        for name in files:
            if name.endswith('.jpg'):
                print("Found")
                SourceFolder = os.path.join(root,name)
                shutil.copy2(SourceFolder, TargetFolder) #copies csv to new folder