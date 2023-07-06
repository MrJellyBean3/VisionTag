from pickletools  import pybytes
import shutil
import socket
import select
import time
import datetime
import os
import multiprocessing
import threading




#setup the file paths
inferencing_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(inferencing_path, os.pardir))
p1_ims_dir=parent_dir+r"\incoming_images\p1"
p2_ims_dir=parent_dir+r"\incoming_images\p2"
comm_file_namep1=parent_dir+r"\inferencing_folder\commp1.txt"
comm_file_namep2=parent_dir+r"\inferencing_folder\commp2.txt"
saving_path=parent_dir+r"\saved_data\raw_images"


def main():
    print("starting main")
    start_server(p1_ims_dir, p2_ims_dir, comm_file_namep1, comm_file_namep2)

def clear_folders():
    global  p2_ims_dir, p1_ims_dir, saving_path

    #get time and make a folder with that name
    timestr = time.strftime("%m-%d-%Y-%H-%M-%S")
    backup_folder=saving_path+"\\"+timestr

    #make the backup folders and copy the images into those folders
    os.mkdir(backup_folder)
    p1_save=backup_folder + "\\" + "p1"
    p2_save=backup_folder + "\\" + "p2"
    shutil.copytree(p1_ims_dir,p1_save)
    shutil.copytree(p2_ims_dir,p2_save)
    
    try:
        for f_name in os.listdir(p2_ims_dir):
            try:
                os.remove(str(p2_ims_dir)+"/"+str(f_name))
            except:
                pass
        for f_name in os.listdir(p1_ims_dir):
            try:
                os.remove(str(p1_ims_dir)+"/"+str(f_name))
            except:
                pass
        print("done clearing")
    except Exception as e:
        print("did not clear error: ",e)


def conn_comm(conn_local, dir_images,comm_file_name, audio_bool):
    clear_folders()

    last_send=time.time()
    comm_string="1,100"

    got_data=False
    im_data=b""
    extra_data=b""
    data_overflow=False

    hit_prev=False
    kill_prev=False

    count_variable=0
    print("Starting connection with: ", conn_local)
    while True:
        ready=select.select([conn_local],[],[],0.01)
        try:
            #This is responsible for sending data to the raspberry pis.
            if ((time.time()-last_send)>0.01):
                comm_file=open(comm_file_name,'r+')
                comm_string=""
                comm_string+=comm_file.readline()
                if(len(comm_string)>0):
                    if(comm_string[0]=='1'):#send hit marker
                        if hit_prev:
                            comm_string='0'+comm_string[1:]
                            comm_file.seek(0)
                            comm_file.write(comm_string)
                        hit_prev=True
                    else:
                        hit_prev=False

                    if(comm_string[0]=='2'):#send kill notification
                        if kill_prev:
                            comm_string='0'+comm_string[1:]
                            comm_file.seek(0)
                            comm_file.write(comm_string)
                        kill_prev=True
                    else:
                        kill_prev=False
                comm_file.close()
                conn_local.send(comm_string.encode("ascii"))#send the player data
                last_send=time.time()



            
            #This recieves the image data from the raspberry pis.
            ready=select.select([conn_local],[],[],0.01)
            if ready[0]:
                in_data=conn_local.recv(1024*5)
                if in_data:
                    chunks=[]
                    if data_overflow:
                        chunks.append(excess_data)
                    in_start_time=time.time()

                    #Loop to grab all image data
                    while in_data!=b"next":
                        if (len(in_data)>1):
                            if (in_data.find(b"next")>0):
                                chunks.append(in_data[0:in_data.find(b"next")])
                                #When the data length exceeds the "next" stop signal the overflow is saved for the next loop
                                if (len(in_data)-in_data.find(b"next"))>4:
                                    excess_data=in_data[in_data.find(b"next")+4:]
                                    data_overflow=True
                                else:
                                    data_overflow=False
                                    excess_data=b""
                                break
                            else:
                                chunks.append(in_data)
                                in_data=conn_local.recv(1024*5)

                    print("len data: ",len(in_data))
                    print("image recieved")

                    #Joins the image data and saves the file
                    im_data = b''.join(chunks)
                    count_variable+=1
                    file_test=open(str(dir_images)+"/"+"incoming_img"+str(count_variable)+".JPG", 'wb')
                    file_test.write(im_data)
                    file_test.close()
        except Exception as e:
            print("\nException: ", e)
            break
    print("Connection Ended")
    return(0)


def start_server(p1_ims_dir, p2_ims_dir, comm_file_namep1, comm_file_namep2):
    print("Starting Server")
    my_ip = "192.168.1.2"
    s1 = socket.socket()
    s1.bind((my_ip, 2345))
    print("listening")
    s1.listen(1)
    print("starting")
    while True:
        time.sleep(0.05)
        conn, addr = s1.accept()
        print("Connected to:", addr)
        if addr[0] != '192.168.1.5':  # Device 1 IP
            t1 = threading.Thread(target=conn_comm, args=(conn, p1_ims_dir, comm_file_namep1, False))
            print("Starting connection to Device 1")
            t1.start()
        elif addr[0] == '192.168.1.5':  # Device 2 IP
            t2 = threading.Thread(target=conn_comm, args=(conn, p2_ims_dir, comm_file_namep2, False))
            print("Starting connection to Device 2")
            t2.start()


if __name__ == "__main__":
    main()
  
