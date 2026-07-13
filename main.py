import argparse, os, cv2

#get video and output paths
parser = argparse.ArgumentParser(prog="videos-to-images")
parser.add_argument("videopath", type=str)
parser.add_argument("outputpath", type=str)
args = parser.parse_args()

video_path = args.videopath
output_path = args.outputpath


if os.path.exists(video_path):
        #get individual video path
        videos = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]

        for video in videos:
            frame_count = 1
            #get video name
            video_name = os.path.splitext(video)
            video_name = video_name[0]

            #name video directory path
            video_directory = os.path.join(output_path, video_name)

            #create video directory path
            os.makedirs(video_directory, exist_ok=True)
        
            #capture video frames
            file_path = os.path.join(video_path, video)
            cap = cv2.VideoCapture(file_path)

            #if video couldn't be opened, discontinue process
            if not cap.isOpened():
                print("Error: Cannot open video file")
                exit()
            
            while True:
                #read the next frame from the video
                success, frame = cap.read()
            
                #If success is False, the video has ended
                if not success:
                    break
                
                #format with zero-padded timestap
                frame_filename = os.path.join(video_directory, f"{frame_count:07d}.jpg")

                #save as JPEG file
                cv2.imwrite(frame_filename, frame)

                frame_count += 1

            cap.release()