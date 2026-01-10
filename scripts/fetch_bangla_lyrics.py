"""
Fetch Bangla lyrics from public sources and add to metadata.csv
Respects robots.txt and includes rate limiting for ethical scraping.
"""

import os
import csv
import time
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
try:
    import langid  # type: ignore
    _LANGID_AVAILABLE = True
except Exception:
    # Optional dependency. If not installed, fall back to a lightweight heuristic.
    langid = None  # type: ignore
    _LANGID_AVAILABLE = False
from urllib.parse import urljoin, quote
from urllib.robotparser import RobotFileParser

# Configuration
BASE_URL = "https://www.lyricsmint.com/bangla"  # Example - adjust to actual source
OUTPUT_CSV = "data/metadata.csv"
DELAY_SECONDS = 2  # Respectful crawling delay
MAX_SONGS = 50  # Limit for initial collection
MIN_LYRICS_LENGTH = 200  # Minimum character count
USER_AGENT = "Mozilla/5.0 (Educational Research Bot)"

class BanglaLyricsScraper:
    def __init__(self, base_url, delay=2):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.robot_parser = RobotFileParser()
        
    def check_robots_txt(self, url):
        """Check if crawling is allowed by robots.txt"""
        try:
            robots_url = urljoin(url, '/robots.txt')
            self.robot_parser.set_url(robots_url)
            self.robot_parser.read()
            return self.robot_parser.can_fetch(USER_AGENT, url)
        except:
            # If can't read robots.txt, assume crawling is not allowed
            return False
    
    def clean_lyrics(self, text):
        """Clean and normalize lyrics text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common ads/footer text
        text = re.sub(r'(advertisement|ads by|powered by|copyright).*', '', text, flags=re.IGNORECASE)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def detect_language(self, text):
        """Detect if text is Bangla.

        Prefers `langid` when available; otherwise uses a simple Bengali-script
        character ratio heuristic.
        """
        if not text:
            return None, 0.0

        if _LANGID_AVAILABLE and langid is not None:
            try:
                lang, confidence = langid.classify(text)
                return lang, float(confidence)
            except Exception:
                return None, 0.0

        # Fallback heuristic: detect Bengali (Bangla) Unicode block characters.
        # Bengali block: U+0980..U+09FF
        bengali_chars = re.findall(r"[\u0980-\u09FF]", text)
        letters = re.findall(r"[A-Za-z\u0980-\u09FF]", text)
        if not letters:
            return None, 0.0

        ratio = len(bengali_chars) / max(1, len(letters))
        # Heuristic thresholds: tune if needed.
        if ratio >= 0.20:
            return "bn", ratio
        if ratio <= 0.02:
            return "en", 1.0 - ratio
        return None, ratio
    
    def sanitize_id(self, title, artist=""):
        """Create sanitized ID from title and artist"""
        text = f"{artist}_{title}" if artist else title
        # Remove special characters, replace spaces with underscores
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text[:100]  # Limit length
    
    def fetch_song_list(self):
        """
        Fetch list of Bangla songs from source
        NOTE: This is a template - adjust selectors based on actual website structure
        """
        songs = []
        
        try:
            # Example: Fetch from a list page
            list_url = f"{self.base_url}/songs-list"
            
            if not self.check_robots_txt(list_url):
                print(f"⚠️  Crawling not allowed by robots.txt: {list_url}")
                return []
            
            print(f"📋 Fetching song list from: {list_url}")
            response = self.session.get(list_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ADJUST THESE SELECTORS BASED ON ACTUAL WEBSITE STRUCTURE
            # Example: Find all song links
            song_links = soup.select('.song-item a.song-link')[:MAX_SONGS]
            
            for link in song_links:
                song_url = urljoin(self.base_url, link.get('href', ''))
                title = link.get_text(strip=True)
                
                if song_url and title:
                    songs.append({
                        'title': title,
                        'url': song_url
                    })
            
            print(f"✅ Found {len(songs)} songs")
            
        except Exception as e:
            print(f"❌ Error fetching song list: {e}")
        
        return songs
    
    def fetch_lyrics(self, song_url, title):
        """
        Fetch lyrics for a specific song
        NOTE: Adjust selectors based on actual website structure
        """
        try:
            if not self.check_robots_txt(song_url):
                print(f"⚠️  Skipping (robots.txt): {title}")
                return None
            
            print(f"🎵 Fetching: {title}")
            time.sleep(self.delay)  # Respectful delay
            
            response = self.session.get(song_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ADJUST THESE SELECTORS BASED ON ACTUAL WEBSITE STRUCTURE
            # Example selectors - modify for actual site
            lyrics_container = soup.select_one('.lyrics-content, .song-lyrics, #lyrics')
            artist_elem = soup.select_one('.artist-name, .song-artist')
            
            if not lyrics_container:
                print(f"⚠️  No lyrics found: {title}")
                return None
            
            lyrics_text = lyrics_container.get_text(separator='\n', strip=True)
            lyrics_clean = self.clean_lyrics(lyrics_text)
            
            # Validate length
            if len(lyrics_clean) < MIN_LYRICS_LENGTH:
                print(f"⚠️  Lyrics too short: {title}")
                return None
            
            # Detect language
            lang, confidence = self.detect_language(lyrics_clean)
            
            if lang not in ['bn', 'en']:
                print(f"⚠️  Language not Bangla/English ({lang}): {title}")
                return None
            
            artist = artist_elem.get_text(strip=True) if artist_elem else "Unknown"
            song_id = self.sanitize_id(title, artist)
            
            return {
                'id': song_id,
                'audio_path': '',  # Empty for now - can be filled later
                'lyrics': lyrics_clean,
                'language': lang,
                'label': lang,
                'source_url': song_url,
                'artist': artist,
                'title': title
            }
            
        except Exception as e:
            print(f"❌ Error fetching {title}: {e}")
            return None
    
    def scrape_bangla_songs(self):
        """Main scraping workflow"""
        print("=" * 60)
        print("🇧🇩 Bangla Lyrics Scraper")
        print("=" * 60)
        
        # Fetch song list
        songs = self.fetch_song_list()
        
        if not songs:
            print("\n⚠️  No songs found. Please check:")
            print("   1. The BASE_URL is correct and accessible")
            print("   2. The CSS selectors match the actual website")
            print("   3. You have internet connection")
            return []
        
        # Fetch lyrics for each song
        results = []
        for i, song in enumerate(songs, 1):
            print(f"\n[{i}/{len(songs)}]", end=" ")
            lyrics_data = self.fetch_lyrics(song['url'], song['title'])
            
            if lyrics_data:
                results.append(lyrics_data)
        
        print(f"\n\n✅ Successfully fetched {len(results)} songs")
        return results


def append_to_metadata_csv(new_songs, csv_path):
    """Append new songs to metadata.csv"""
    
    if not new_songs:
        print("\n⚠️  No songs to append")
        return
    
    # Check if CSV exists and has data
    file_exists = os.path.exists(csv_path)
    
    # Read existing IDs to avoid duplicates
    existing_ids = set()
    if file_exists:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_ids = {row['id'] for row in reader}
        except Exception as e:
            print(f"⚠️  Error reading existing CSV: {e}")
    
    # Filter out duplicates
    new_songs_filtered = [s for s in new_songs if s['id'] not in existing_ids]
    
    if not new_songs_filtered:
        print("\n⚠️  All songs already exist in metadata.csv")
        return
    
    # Append to CSV
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'audio_path', 'lyrics', 'language', 'label']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if new file
            if not file_exists or os.path.getsize(csv_path) == 0:
                writer.writeheader()
            
            for song in new_songs_filtered:
                writer.writerow({
                    'id': song['id'],
                    'audio_path': song['audio_path'],
                    'lyrics': song['lyrics'],
                    'language': song['language'],
                    'label': song['label']
                })
        
        print(f"\n✅ Added {len(new_songs_filtered)} new songs to {csv_path}")
        print(f"   ({len(new_songs) - len(new_songs_filtered)} duplicates skipped)")
        
    except Exception as e:
        print(f"❌ Error writing to CSV: {e}")


def main():
    """Main execution"""
    print("\n⚠️  IMPORTANT: Manual Configuration Required!")
    print("=" * 60)
    print("This script is a TEMPLATE. Before running:")
    print("1. Find a legitimate Bangla lyrics source (check license/ToS)")
    print("2. Update BASE_URL with the actual website")
    print("3. Inspect the website HTML and update CSS selectors:")
    print("   - Song list selectors in fetch_song_list()")
    print("   - Lyrics content selectors in fetch_lyrics()")
    print("4. Install dependencies: pip install requests beautifulsoup4 langid")
    print("5. Verify the site's robots.txt allows crawling")
    print("=" * 60)
    
    proceed = input("\n✅ Have you configured the script? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("\n⏸️  Please configure the script first, then run again.")
        return
    
    # Initialize scraper
    scraper = BanglaLyricsScraper(BASE_URL, delay=DELAY_SECONDS)
    
    # Scrape songs
    bangla_songs = scraper.scrape_bangla_songs()
    
    # Append to metadata.csv
    if bangla_songs:
        append_to_metadata_csv(bangla_songs, OUTPUT_CSV)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Summary:")
        print("=" * 60)
        
        lang_count = {}
        for song in bangla_songs:
            lang = song['language']
            lang_count[lang] = lang_count.get(lang, 0) + 1
        
        for lang, count in lang_count.items():
            lang_name = "Bangla" if lang == "bn" else "English"
            print(f"   {lang_name} ({lang}): {count} songs")
        
        print("\n✅ Dataset updated successfully!")
        print(f"   Check: {OUTPUT_CSV}")
    else:
        print("\n❌ No songs were collected. Check the configuration and try again.")


if __name__ == "__main__":
    main()
