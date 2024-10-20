# Torrent Downloader and Streamer

This Python script allows you to download torrents using magnet links and stream the downloaded content using FFmpeg.

## Prerequisites

- Python 3
- libtorrent
- FFmpeg
- tqdm

## Installation

1. Install the necessary libraries:
   ```
   apt-get install -y python3-libtorrent ffmpeg
   pip install tqdm
   ```

## Usage

1. Run the script and enter the magnet link when prompted.
2. The script will download the torrent and display progress information.
3. Once the download is complete, it will attempt to stream the content using FFmpeg.

## Streaming

To play the streamed content:
- Use VLC player and open the network stream: `udp://@127.0.0.1:1234`

Alternatively (recommended method):
- Click the "download" button
- Use an IDM (Internet Download Manager) extension or download directly
- In your download history, copy the full HTTPS link
- Paste the link into VLC's network streaming option

This alternative method provides a more direct way to play the content and should work well.

## Note

This script is for educational purposes only. Ensure you have the right to download and stream the content.

## Disclaimer

Be aware of the legal implications of downloading and sharing copyrighted material in your jurisdiction.
## CODE

```
# Install necessary libraries
!apt-get install -y python3-libtorrent ffmpeg
!pip install tqdm

import libtorrent as lt
import time
import os
from tqdm import tqdm

# Function to download the torrent using a magnet link
def download_torrent(magnet_link, download_path='/content/'):
    # Create a session with optimized settings for faster downloading
    ses = lt.session()
    ses.listen_on(6881, 6891)
    
    # Optimize session settings for better speed
    settings = ses.get_settings()
    settings['download_rate_limit'] = 0  # Unlimited download rate
    settings['upload_rate_limit'] = 0    # Unlimited upload rate
    settings['active_downloads'] = 10    # Number of active downloads
    settings['active_limit'] = 100       # Maximum number of connections
    ses.apply_settings(settings)
    
    # Add the magnet link
    params = lt.add_torrent_params()
    params.save_path = download_path
    params.url = magnet_link
    
    print("Adding torrent from magnet link...")
    handle = ses.add_torrent(params)
    
    print("Downloading metadata...")
    
    # Wait for the torrent metadata to download
    while not handle.has_metadata():
        time.sleep(1)
    
    print("Metadata downloaded, starting torrent download...")
    
    # Set up a progress bar
    progress_bar = tqdm(total=100, unit='%')

    # Monitor download progress
    while handle.status().state != lt.torrent_status.seeding:
        status = handle.status()
        
        progress = status.progress * 100  # Progress in percentage
        download_rate = status.download_rate / 1000  # Download speed in kB/s
        upload_rate = status.upload_rate / 1000  # Upload speed in kB/s
        num_peers = status.num_peers  # Number of connected peers

        # Calculate estimated time (ETA)
        if download_rate > 0:
            remaining = (status.total_wanted - status.total_wanted_done) / download_rate / 1000
            eta = time.strftime('%H:%M:%S', time.gmtime(remaining))
        else:
            eta = '∞'
        
        # Update progress bar
        progress_bar.n = progress
        progress_bar.set_postfix({
            'Download Speed': f'{download_rate:.2f} kB/s',
            'Upload Speed': f'{upload_rate:.2f} kB/s',
            'Peers': num_peers,
            'ETA': eta
        })
        progress_bar.update(0)
        
        time.sleep(1)  # Check the status every second

    # Complete the progress bar when done
    progress_bar.n = 100
    progress_bar.close()
    
    print("Download complete!")
    return download_path + handle.name()

# Prompt user to enter the magnet link
magnet_link = input("Enter the torrent magnet link: ")

# Download the torrent file
downloaded_file = download_torrent(magnet_link)

# Play the movie using ffmpeg
print(f"Streaming {downloaded_file}...")

# Command to stream using ffmpeg
os.system(f'ffmpeg -i "{downloaded_file}" -f mpegts udp://127.0.0.1:1234')

# To play it locally, use the following command in your VLC player:
# "udp://@127.0.0.1:1234"

# the better way to play directly is to  click down "download"   i recommend use IDM  extension , or just download direct and in your downloading History , copy the full https link,, and paste it in VLC , netwoking streaming , it will work 


```
