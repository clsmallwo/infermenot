from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


LETTERS = ("A", "B", "C", "D")
SOURCE = "synthetic/music_song_history_v1"


@dataclass(frozen=True)
class MusicMCQ:
    id: str
    source: str
    split: str
    question: str
    choices: dict[str, str]
    answer: str


SONGS = [
    {"title": "Strange Fruit", "artist": "Billie Holiday", "year": "1939", "album": "", "style": "jazz protest song"},
    {"title": "Rock Around the Clock", "artist": "Bill Haley & His Comets", "year": "1954", "album": "", "style": "rock and roll"},
    {"title": "Hound Dog", "artist": "Elvis Presley", "year": "1956", "album": "", "style": "rock and roll"},
    {"title": "Johnny B. Goode", "artist": "Chuck Berry", "year": "1958", "album": "", "style": "rock and roll"},
    {"title": "What'd I Say", "artist": "Ray Charles", "year": "1959", "album": "", "style": "rhythm and blues"},
    {"title": "So What", "artist": "Miles Davis", "year": "1959", "album": "Kind of Blue", "style": "modal jazz"},
    {"title": "Take Five", "artist": "The Dave Brubeck Quartet", "year": "1959", "album": "Time Out", "style": "cool jazz"},
    {"title": "Giant Steps", "artist": "John Coltrane", "year": "1960", "album": "Giant Steps", "style": "hard bop"},
    {"title": "Be My Baby", "artist": "The Ronettes", "year": "1963", "album": "", "style": "girl group pop"},
    {"title": "A Change Is Gonna Come", "artist": "Sam Cooke", "year": "1964", "album": "Ain't That Good News", "style": "soul"},
    {"title": "My Girl", "artist": "The Temptations", "year": "1964", "album": "The Temptations Sing Smokey", "style": "Motown soul"},
    {"title": "Like a Rolling Stone", "artist": "Bob Dylan", "year": "1965", "album": "Highway 61 Revisited", "style": "folk rock"},
    {"title": "A Love Supreme, Part I: Acknowledgement", "artist": "John Coltrane", "year": "1965", "album": "A Love Supreme", "style": "spiritual jazz"},
    {"title": "Good Vibrations", "artist": "The Beach Boys", "year": "1966", "album": "", "style": "psychedelic pop"},
    {"title": "God Only Knows", "artist": "The Beach Boys", "year": "1966", "album": "Pet Sounds", "style": "baroque pop"},
    {"title": "Respect", "artist": "Aretha Franklin", "year": "1967", "album": "I Never Loved a Man the Way I Love You", "style": "soul"},
    {"title": "Light My Fire", "artist": "The Doors", "year": "1967", "album": "The Doors", "style": "psychedelic rock"},
    {"title": "Purple Haze", "artist": "The Jimi Hendrix Experience", "year": "1967", "album": "Are You Experienced", "style": "psychedelic rock"},
    {"title": "All Along the Watchtower", "artist": "The Jimi Hendrix Experience", "year": "1968", "album": "Electric Ladyland", "style": "psychedelic rock"},
    {"title": "Whole Lotta Love", "artist": "Led Zeppelin", "year": "1969", "album": "Led Zeppelin II", "style": "hard rock"},
    {"title": "Bridge Over Troubled Water", "artist": "Simon & Garfunkel", "year": "1970", "album": "Bridge Over Troubled Water", "style": "folk pop"},
    {"title": "Let It Be", "artist": "The Beatles", "year": "1970", "album": "Let It Be", "style": "rock"},
    {"title": "American Pie", "artist": "Don McLean", "year": "1971", "album": "American Pie", "style": "folk rock"},
    {"title": "Imagine", "artist": "John Lennon", "year": "1971", "album": "Imagine", "style": "soft rock"},
    {"title": "Let's Stay Together", "artist": "Al Green", "year": "1971", "album": "Let's Stay Together", "style": "soul"},
    {"title": "Stairway to Heaven", "artist": "Led Zeppelin", "year": "1971", "album": "Led Zeppelin IV", "style": "rock"},
    {"title": "Superstition", "artist": "Stevie Wonder", "year": "1972", "album": "Talking Book", "style": "funk"},
    {"title": "What's Going On", "artist": "Marvin Gaye", "year": "1971", "album": "What's Going On", "style": "soul"},
    {"title": "Born to Run", "artist": "Bruce Springsteen", "year": "1975", "album": "Born to Run", "style": "heartland rock"},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "year": "1975", "album": "A Night at the Opera", "style": "progressive rock"},
    {"title": "Hotel California", "artist": "Eagles", "year": "1976", "album": "Hotel California", "style": "rock"},
    {"title": "I Will Survive", "artist": "Gloria Gaynor", "year": "1978", "album": "Love Tracks", "style": "disco"},
    {"title": "London Calling", "artist": "The Clash", "year": "1979", "album": "London Calling", "style": "punk rock"},
    {"title": "Rapper's Delight", "artist": "The Sugarhill Gang", "year": "1979", "album": "", "style": "hip hop"},
    {"title": "Billie Jean", "artist": "Michael Jackson", "year": "1982", "album": "Thriller", "style": "pop"},
    {"title": "The Message", "artist": "Grandmaster Flash and the Furious Five", "year": "1982", "album": "The Message", "style": "hip hop"},
    {"title": "Blue Monday", "artist": "New Order", "year": "1983", "album": "", "style": "synth-pop"},
    {"title": "Like a Virgin", "artist": "Madonna", "year": "1984", "album": "Like a Virgin", "style": "dance-pop"},
    {"title": "When Doves Cry", "artist": "Prince", "year": "1984", "album": "Purple Rain", "style": "funk rock"},
    {"title": "Take On Me", "artist": "a-ha", "year": "1985", "album": "Hunting High and Low", "style": "synth-pop"},
    {"title": "Walk This Way", "artist": "Run-D.M.C.", "year": "1986", "album": "Raising Hell", "style": "rap rock"},
    {"title": "Sweet Child o' Mine", "artist": "Guns N' Roses", "year": "1987", "album": "Appetite for Destruction", "style": "hard rock"},
    {"title": "Fight the Power", "artist": "Public Enemy", "year": "1989", "album": "", "style": "political hip hop"},
    {"title": "Losing My Religion", "artist": "R.E.M.", "year": "1991", "album": "Out of Time", "style": "alternative rock"},
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "year": "1991", "album": "Nevermind", "style": "grunge"},
    {"title": "Creep", "artist": "Radiohead", "year": "1992", "album": "Pablo Honey", "style": "alternative rock"},
    {"title": "Nuthin' but a 'G' Thang", "artist": "Dr. Dre", "year": "1992", "album": "The Chronic", "style": "G-funk"},
    {"title": "Juicy", "artist": "The Notorious B.I.G.", "year": "1994", "album": "Ready to Die", "style": "East Coast hip hop"},
    {"title": "Wonderwall", "artist": "Oasis", "year": "1995", "album": "(What's the Story) Morning Glory?", "style": "Britpop"},
    {"title": "Bitter Sweet Symphony", "artist": "The Verve", "year": "1997", "album": "Urban Hymns", "style": "Britpop"},
    {"title": "Doo Wop (That Thing)", "artist": "Lauryn Hill", "year": "1998", "album": "The Miseducation of Lauryn Hill", "style": "neo soul"},
    {"title": "No Scrubs", "artist": "TLC", "year": "1999", "album": "FanMail", "style": "R&B"},
    {"title": "Harder, Better, Faster, Stronger", "artist": "Daft Punk", "year": "2001", "album": "Discovery", "style": "French house"},
    {"title": "Crazy in Love", "artist": "Beyonce", "year": "2003", "album": "Dangerously in Love", "style": "R&B pop"},
    {"title": "Hey Ya!", "artist": "OutKast", "year": "2003", "album": "Speakerboxxx/The Love Below", "style": "funk pop"},
    {"title": "Maps", "artist": "Yeah Yeah Yeahs", "year": "2003", "album": "Fever to Tell", "style": "indie rock"},
    {"title": "Seven Nation Army", "artist": "The White Stripes", "year": "2003", "album": "Elephant", "style": "garage rock"},
    {"title": "Rehab", "artist": "Amy Winehouse", "year": "2006", "album": "Back to Black", "style": "soul"},
    {"title": "Paper Planes", "artist": "M.I.A.", "year": "2007", "album": "Kala", "style": "alternative hip hop"},
    {"title": "Single Ladies (Put a Ring on It)", "artist": "Beyonce", "year": "2008", "album": "I Am... Sasha Fierce", "style": "dance-pop"},
    {"title": "Bad Romance", "artist": "Lady Gaga", "year": "2009", "album": "The Fame Monster", "style": "electropop"},
    {"title": "Rolling in the Deep", "artist": "Adele", "year": "2010", "album": "21", "style": "soul pop"},
    {"title": "Runaway", "artist": "Kanye West", "year": "2010", "album": "My Beautiful Dark Twisted Fantasy", "style": "hip hop"},
    {"title": "Get Lucky", "artist": "Daft Punk", "year": "2013", "album": "Random Access Memories", "style": "disco funk"},
    {"title": "Royals", "artist": "Lorde", "year": "2013", "album": "Pure Heroine", "style": "art pop"},
    {"title": "Uptown Funk", "artist": "Mark Ronson", "year": "2014", "album": "Uptown Special", "style": "funk pop"},
    {"title": "Alright", "artist": "Kendrick Lamar", "year": "2015", "album": "To Pimp a Butterfly", "style": "hip hop"},
    {"title": "Formation", "artist": "Beyonce", "year": "2016", "album": "Lemonade", "style": "R&B"},
    {"title": "HUMBLE.", "artist": "Kendrick Lamar", "year": "2017", "album": "DAMN.", "style": "hip hop"},
    {"title": "Shape of You", "artist": "Ed Sheeran", "year": "2017", "album": "Divide", "style": "pop"},
    {"title": "Bad Guy", "artist": "Billie Eilish", "year": "2019", "album": "When We All Fall Asleep, Where Do We Go?", "style": "electropop"},
    {"title": "Blinding Lights", "artist": "The Weeknd", "year": "2019", "album": "After Hours", "style": "synth-pop"},
    {"title": "Levitating", "artist": "Dua Lipa", "year": "2020", "album": "Future Nostalgia", "style": "disco pop"},
    {"title": "Drivers License", "artist": "Olivia Rodrigo", "year": "2021", "album": "SOUR", "style": "pop ballad"},
    {"title": "As It Was", "artist": "Harry Styles", "year": "2022", "album": "Harry's House", "style": "synth-pop"},
]


HISTORY_FACTS = [
    ("Which composer is strongly associated with the Baroque era and the Brandenburg Concertos?", "Johann Sebastian Bach", ["Franz Schubert", "Claude Debussy", "Igor Stravinsky", "John Cage"]),
    ("Which composer wrote The Magic Flute and is a central figure of the Classical period?", "Wolfgang Amadeus Mozart", ["Richard Wagner", "Gustav Mahler", "Arnold Schoenberg", "Claudio Monteverdi"]),
    ("Which composer wrote Symphony No. 9 and helped bridge the Classical and Romantic eras?", "Ludwig van Beethoven", ["Antonio Vivaldi", "Scott Joplin", "Duke Ellington", "Philip Glass"]),
    ("Who is often called the father of the symphony and the string quartet?", "Joseph Haydn", ["Hector Berlioz", "Robert Johnson", "George Gershwin", "Brian Eno"]),
    ("Which medieval composer is known for sacred monophonic chant and visionary writings?", "Hildegard of Bingen", ["Clara Schumann", "Nina Simone", "Patti Smith", "Sister Rosetta Tharpe"]),
    ("Which Renaissance composer is closely linked with smooth sacred polyphony and the Counter-Reformation?", "Giovanni Pierluigi da Palestrina", ["Dmitri Shostakovich", "Fats Domino", "Steve Reich", "Bela Bartok"]),
    ("Which term describes the repeating bass pattern common in many Baroque pieces?", "basso continuo", ["backbeat", "breakbeat", "blue note", "twelve-tone row"]),
    ("Which form is built from imitative counterpoint, with a subject entering in different voices?", "fugue", ["rondo", "verse-chorus form", "shuffle", "breakdown"]),
    ("Which large-scale form commonly includes exposition, development, and recapitulation?", "sonata-allegro form", ["twelve-bar blues", "strophic form", "call and response", "tintinnabuli"]),
    ("Opera first developed as a major art form in which place and period?", "late Renaissance Italy", ["medieval Scandinavia", "nineteenth-century Brazil", "ancient Mesopotamia", "postwar Detroit"]),
    ("Which early opera by Monteverdi is a landmark of the genre?", "L'Orfeo", ["The Rite of Spring", "Kind of Blue", "Aida", "West Side Story"]),
    ("Which Baroque composer wrote The Four Seasons?", "Antonio Vivaldi", ["Giacomo Puccini", "Charles Ives", "Herbie Hancock", "Kraftwerk"]),
    ("Which oratorio by Handel is famous for its Hallelujah chorus?", "Messiah", ["Carmina Burana", "The Planets", "Pierrot lunaire", "Rhapsody in Blue"]),
    ("Which period emphasized expressive chromatic harmony, expanded orchestras, and personal emotion?", "Romantic era", ["Ars nova", "bebop era", "minimalism", "early punk"]),
    ("Which composer is commonly associated with musical Impressionism?", "Claude Debussy", ["Hank Williams", "Dizzy Gillespie", "Berry Gordy", "Brian Wilson"]),
    ("Ragtime piano is especially associated with which composer?", "Scott Joplin", ["Dolly Parton", "Charlie Parker", "Robert Schumann", "Terry Riley"]),
    ("Which musical feature is central to the blues?", "blue notes", ["serial rows", "basso continuo", "prepared piano bolts", "isorhythm only"]),
    ("Which city is widely associated with the early development of jazz?", "New Orleans", ["Vienna", "Liverpool", "Seattle", "Berlin"]),
    ("Bebop is closely associated with which pair of musicians?", "Charlie Parker and Dizzy Gillespie", ["Lennon and McCartney", "Hall and Oates", "Simon and Garfunkel", "Jagger and Richards"]),
    ("Which record label founded by Berry Gordy became central to the Detroit soul sound?", "Motown", ["Stax", "Blue Note", "Sub Pop", "Sun Records"]),
    ("Which Jamaican style is known for offbeat guitar or keyboard accents called the skank?", "reggae", ["bebop", "ragtime", "baroque suite", "grunge"]),
    ("Which DJ is widely cited as a foundational figure in early Bronx hip hop?", "DJ Kool Herc", ["Brian Eno", "Benny Goodman", "Phil Spector", "Pete Seeger"]),
    ("Hip hop is commonly traced to block parties in which New York borough?", "the Bronx", ["Queens", "Manhattan", "Staten Island", "Brooklyn"]),
    ("Which electronic dance style is strongly associated with Chicago in the 1980s?", "house music", ["bluegrass", "highlife", "krautrock", "doo-wop"]),
    ("Which electronic dance style is strongly associated with Detroit in the 1980s?", "techno", ["honky-tonk", "operetta", "barbershop", "calypso"]),
    ("Grunge is most strongly associated with which city in the early 1990s?", "Seattle", ["Memphis", "Nashville", "Atlanta", "Minneapolis"]),
    ("Which record label is strongly associated with early Seattle grunge releases?", "Sub Pop", ["Chess Records", "Motown", "Def Jam", "Blue Note"]),
    ("Which label was central to Southern soul from Memphis?", "Stax Records", ["ECM Records", "Factory Records", "Sire Records", "4AD"]),
    ("Which label is strongly associated with classic jazz recordings by artists such as Art Blakey and Herbie Hancock?", "Blue Note", ["Stax", "Factory", "Rough Trade", "Tamla"]),
    ("Which production figure is known for the Wall of Sound?", "Phil Spector", ["Rick Rubin", "Quincy Jones", "Sam Phillips", "George Martin"]),
    ("Which producer worked extensively with The Beatles and is often called their producer?", "George Martin", ["Teo Macero", "Brian Eno", "Timbaland", "Max Martin"]),
    ("Which instrument family does the saxophone belong to?", "woodwinds", ["brass", "strings", "pitched percussion", "membranophones"]),
    ("Which instrument is central to bluegrass banjo rolls?", "banjo", ["oboe", "theremin", "timpani", "harpsichord"]),
    ("Which rhythm-and-blues label helped launch Elvis Presley's early recordings?", "Sun Records", ["Sub Pop", "Def Jam", "Island Records", "Warp"]),
    ("Which genre grew from Jamaican sound-system culture and influenced hip hop DJ practice?", "dub", ["chamber opera", "ragtime", "bebop", "serialism"]),
    ("Which term means a short repeated musical phrase that can anchor a song?", "riff", ["libretto", "cadenza", "aria", "recitative"]),
    ("Which term describes a singer improvising nonsense syllables in jazz?", "scat singing", ["sprechstimme", "melisma only", "pizzicato", "tremolo picking"]),
    ("Which term describes alternating phrases between a leader and a group?", "call and response", ["through-composition", "equal temperament", "prepared piano", "tone row"]),
    ("Which term names the main repeated hook or chorus-like section in many popular songs?", "refrain", ["bridge pickup", "coda only", "development section", "ground bass"]),
    ("Which genre is associated with Nashville, honky-tonk, and pedal steel guitar?", "country music", ["krautrock", "bebop", "samba", "minimalism"]),
    ("Which genre is associated with Jamaica, Rastafari themes, and Bob Marley?", "reggae", ["ragtime", "opera buffa", "cool jazz", "glam rock"]),
    ("Which musical movement often uses slowly changing repeated patterns?", "minimalism", ["bebop", "doowop", "motet", "ragtime"]),
    ("Which composer is a major figure in American minimalism?", "Steve Reich", ["Robert Johnson", "Giuseppe Verdi", "Clara Schumann", "Buddy Holly"]),
    ("Which composer is known for the ballet The Rite of Spring?", "Igor Stravinsky", ["Bessie Smith", "Cole Porter", "Kurt Cobain", "Tina Turner"]),
    ("The Rite of Spring is closely linked with which early twentieth-century movement?", "modernism", ["doo-wop", "surf rock", "trap", "bel canto"]),
    ("Which blues musician is famously associated with Delta blues guitar and the song Cross Road Blues?", "Robert Johnson", ["Max Roach", "Bing Crosby", "Eddie Van Halen", "Kate Bush"]),
    ("Which singer is often called the Empress of the Blues?", "Bessie Smith", ["Joni Mitchell", "Ella Fitzgerald", "Diana Ross", "Dusty Springfield"]),
    ("Which singer is often called the Queen of Soul?", "Aretha Franklin", ["Patsy Cline", "Billie Eilish", "Grace Slick", "Celia Cruz"]),
    ("Which Cuban singer is often called the Queen of Salsa?", "Celia Cruz", ["Nico", "Joan Baez", "Sandy Denny", "Roberta Flack"]),
    ("Which band is strongly associated with Liverpool and the British Invasion?", "The Beatles", ["The Ramones", "The Stooges", "Kraftwerk", "R.E.M."]),
    ("Which American label helped popularize early hip hop through releases like Rapper's Delight?", "Sugar Hill Records", ["Deutsche Grammophon", "RCA Victor Red Seal", "Factory Records", "ECM"]),
    ("Which rap label founded by Russell Simmons and Rick Rubin became central to 1980s hip hop?", "Def Jam", ["Sun Records", "Chess Records", "Blue Note", "Motown"]),
    ("Which group is closely associated with the birth of German electronic pop and krautrock influence?", "Kraftwerk", ["The Carter Family", "The Supremes", "The Clash", "ABBA"]),
    ("Which Swedish group became internationally famous after winning Eurovision with Waterloo?", "ABBA", ["The Velvet Underground", "Talking Heads", "Cream", "The Smiths"]),
    ("Which instrument did Jimi Hendrix primarily play?", "electric guitar", ["alto saxophone", "upright bass", "drum kit", "accordion"]),
    ("Which instrument is Stevie Wonder especially known for playing in addition to singing?", "keyboards", ["trombone", "banjo", "bassoon", "mandolin"]),
    ("Which jazz musician is most associated with trumpet and albums such as Kind of Blue?", "Miles Davis", ["John Bonham", "Joni Mitchell", "Eminem", "Johnny Cash"]),
    ("Which jazz musician is most associated with tenor saxophone and A Love Supreme?", "John Coltrane", ["Elvis Costello", "Paul Simon", "Muddy Waters", "Phil Collins"]),
    ("Which production technique layers many instruments into a dense pop arrangement?", "Wall of Sound", ["walking bass", "sonata form", "twelve-tone serialism", "circle singing"]),
    ("Which term describes DJ manipulation of vinyl records to create rhythmic sounds?", "scratching", ["yodeling", "vibrato", "counterpoint", "ostinato"]),
    ("Which drum machine is especially associated with early hip hop, electro, and dance music?", "Roland TR-808", ["Mellotron M400", "Fender Rhodes", "Hammond B-3", "Moog Minimoog"]),
    ("Which keyboard instrument is known for tonewheels and jazz, gospel, and rock organ sounds?", "Hammond B-3", ["Roland TR-808", "Stradivarius violin", "Fairlight CMI", "LinnDrum"]),
]


QUESTION_PREFIXES = (
    "",
    "Music history: ",
    "Popular music study: ",
    "Song catalog question: ",
    "Recording history: ",
)


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def unique_values(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


ARTISTS = unique_values(song["artist"] for song in SONGS)
TITLES = unique_values(song["title"] for song in SONGS)
ALBUMS = unique_values(song["album"] for song in SONGS if song["album"])
YEARS = unique_values(song["year"] for song in SONGS)
STYLES = unique_values(song["style"] for song in SONGS)
HISTORY_ANSWERS = unique_values(answer for _, answer, _ in HISTORY_FACTS)
TITLES_BY_ARTIST = {
    artist: {song["title"] for song in SONGS if song["artist"] == artist}
    for artist in ARTISTS
}


def pick_distractors(correct: str, pool: Sequence[str], rng: random.Random, count: int = 3) -> list[str]:
    candidates = [item for item in pool if item != correct]
    rng.shuffle(candidates)
    return candidates[:count]


def pick_title_distractors_for_artist(artist: str, correct: str, rng: random.Random) -> list[str]:
    associated_titles = TITLES_BY_ARTIST[artist]
    candidates = [title for title in TITLES if title not in associated_titles and title != correct]
    rng.shuffle(candidates)
    return candidates[:3]


def make_row(row_id: int, question: str, correct: str, distractors: Sequence[str], rng: random.Random) -> MusicMCQ:
    options = [correct, *distractors]
    if len(set(options)) < 4:
        raise ValueError(f"Duplicate options for {question!r}: {options!r}")
    rng.shuffle(options)
    choices = dict(zip(LETTERS, options))
    answer = next(letter for letter, value in choices.items() if value == correct)
    return MusicMCQ(
        id=f"music-synth-v1-{row_id:06d}",
        source=SOURCE,
        split="train",
        question=question,
        choices=choices,
        answer=answer,
    )


def song_questions(rng: random.Random) -> Iterable[tuple[str, str, list[str]]]:
    songs = list(SONGS)
    rng.shuffle(songs)
    for song in songs:
        prefix = rng.choice(QUESTION_PREFIXES)
        title = song["title"]
        artist = song["artist"]
        year = song["year"]
        style = song["style"]
        album = song["album"]
        yield (
            f'{prefix}Which artist is associated with the song "{title}"?',
            artist,
            pick_distractors(artist, ARTISTS, rng),
        )
        yield (
            f'{prefix}Which song is associated with {artist}?',
            title,
            pick_title_distractors_for_artist(artist, title, rng),
        )
        yield (
            f'{prefix}In which year was "{title}" released?',
            year,
            pick_distractors(year, YEARS, rng),
        )
        yield (
            f'{prefix}Which style is most closely associated with "{title}"?',
            style,
            pick_distractors(style, STYLES, rng),
        )
        yield (
            f'{prefix}Which song and artist pairing is correct?',
            f'"{title}" - {artist}',
            [f'"{title}" - {wrong}' for wrong in pick_distractors(artist, ARTISTS, rng)],
        )
        yield (
            f'{prefix}Which song and release year pairing is correct?',
            f'"{title}" - {year}',
            [f'"{title}" - {wrong}' for wrong in pick_distractors(year, YEARS, rng)],
        )
        if album:
            yield (
                f'{prefix}Which album includes "{title}"?',
                album,
                pick_distractors(album, ALBUMS, rng),
            )
            yield (
                f'{prefix}Which song and album pairing is correct?',
                f'"{title}" - {album}',
                [f'"{title}" - {wrong}' for wrong in pick_distractors(album, ALBUMS, rng)],
            )


def history_questions(rng: random.Random) -> Iterable[tuple[str, str, list[str]]]:
    facts = list(HISTORY_FACTS)
    rng.shuffle(facts)
    for question, answer, distractors in facts:
        prefix = rng.choice(QUESTION_PREFIXES)
        pool = unique_values([*distractors, *HISTORY_ANSWERS])
        yield (f"{prefix}{question}", answer, pick_distractors(answer, pool, rng))


def generate(path: Path, target_bytes: int, seed: int) -> int:
    rng = random.Random(seed)
    row_id = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        while f.tell() < target_bytes:
            cycle_rng = random.Random(stable_seed(f"{seed}-{row_id}-{f.tell()}"))
            for question, correct, distractors in song_questions(cycle_rng):
                row = make_row(row_id, question, correct, distractors, rng)
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                row_id += 1
                if f.tell() >= target_bytes:
                    return row_id
            for question, correct, distractors in history_questions(cycle_rng):
                row = make_row(row_id, question, correct, distractors, rng)
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                row_id += 1
                if f.tell() >= target_bytes:
                    return row_id
    return row_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local synthetic music MCQ supplement.")
    parser.add_argument("--out", default="data/music_song_history_10mb.jsonl")
    parser.add_argument("--target-mb", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    rows = generate(Path(args.out), int(args.target_mb * 1024 * 1024), args.seed)
    size = Path(args.out).stat().st_size
    print(f"Wrote {rows:,} music MCQs to {args.out} ({size:,} bytes)")


if __name__ == "__main__":
    main()
