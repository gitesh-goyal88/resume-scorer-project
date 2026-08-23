import requests
import re
import json

def fetch_courses(skill: str, limit: int = 3) -> list:
    """
    Performs a real-time HTTP request to YouTube to scrape the top video courses
    for a given missing skill. Extracts data directly from the raw HTML JS objects.
    """
    search_url = f"https://www.youtube.com/results?search_query={skill}+full+course"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return _fallback_links(skill)
            
        html = response.text
        
        # YouTube injects the initial search results into a JS variable called ytInitialData
        data_regex = re.search(r"var ytInitialData = ({.*?});</script>", html)
        if not data_regex:
            return _fallback_links(skill)
            
        json_data = json.loads(data_regex.group(1))
        
        # Traverse the JSON tree to find video renderers
        contents = json_data.get("contents", {}).get("twoColumnSearchResultsRenderer", {}).get("primaryContents", {}).get("sectionListRenderer", {}).get("contents", [])
        
        if not contents:
            return _fallback_links(skill)
            
        items = contents[0].get("itemSectionRenderer", {}).get("contents", [])
        
        results = []
        for item in items:
            if "videoRenderer" in item:
                vid = item["videoRenderer"]
                title = vid.get("title", {}).get("runs", [{}])[0].get("text", "Unknown Title")
                video_id = vid.get("videoId", "")
                channel = vid.get("ownerText", {}).get("runs", [{}])[0].get("text", "Unknown Channel")
                
                if video_id and "course" in title.lower() or "tutorial" in title.lower():
                    results.append({
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "platform": channel
                    })
                    if len(results) == limit:
                        break
                        
        if not results:
            return _fallback_links(skill)
            
        return results
        
    except Exception as e:
        print(f"Scraper error: {e}")
        return _fallback_links(skill)

def _fallback_links(skill: str) -> list:
    """Fallback utility that generates direct search links if scraping fails."""
    return [
        {
            "title": f"Complete {skill} Bootcamp",
            "url": f"https://www.udemy.com/courses/search/?src=ukw&q={skill}",
            "platform": "Udemy Search"
        },
        {
            "title": f"Master {skill} from Scratch",
            "url": f"https://www.coursera.org/search?query={skill}",
            "platform": "Coursera Search"
        },
        {
            "title": f"{skill} Full Course for Beginners",
            "url": f"https://www.youtube.com/results?search_query={skill}+full+course",
            "platform": "YouTube Search"
        }
    ]

if __name__ == "__main__":
    # Test the scraper
    print(fetch_courses("Docker"))
