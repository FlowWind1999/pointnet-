import os
import time
def check(line):
    if(line[-9]=='0'):
        return True
    return False

def apart(path,outpath):
    lst = os.listdir(path)
    for item in lst:
        filepath=os.path.join(path,item)
        print(filepath)
        with open(filepath,'r') as f:
            earth=open(os.path.join(outpath,'0',item),'w+')
            wire=open(os.path.join(outpath,'1',item),'w+')
            file=f.readlines()
            for line in file:
                if line[-9]=='0':
                    earth.write(line)
                else:
                    wire.write(line)
            earth.close()
            wire.close()

if __name__ == "__main__":
    path = r"..\data\ours\15"
    outpath = r"..\data\re"
    start=time.time()
    apart(path,outpath)
    end=time.time()
    print("total used time is ",end-start," s")
