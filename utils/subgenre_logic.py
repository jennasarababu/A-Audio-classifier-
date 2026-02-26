def classify_subgenre(genre, bpm, centroid, zcr, pitch_std):

    if genre == "disco":
        if 120 <= bpm <= 128:
            return "House"
        elif 100 <= bpm < 115:
            return "Tropical House"
        elif 130 <= bpm <= 150:
            return "Trap EDM"
        else:
            return "Dance EDM"

    elif genre == "rock":
        if centroid > 3500 and zcr > 0.12:
            return "Hard Rock"
        elif bpm > 150:
            return "Punk Rock"
        else:
            return "Alternative Rock"

    elif genre == "hiphop":
        if 70 <= bpm <= 90:
            return "Boom Bap"
        elif bpm > 120:
            return "Trap Hip-Hop"
        else:
            return "Lo-Fi Hip-Hop"

    elif genre == "classical":
        if bpm < 80:
            return "Baroque"
        elif pitch_std > 180:
            return "Romantic"
        else:
            return "Modern Classical"

    elif genre == "pop":
        if bpm > 120:
            return "Dance Pop"
        elif pitch_std > 150:
            return "Electro Pop"
        else:
            return "Indie Pop"

    elif genre == "metal":
        if bpm > 170:
            return "Speed Metal"
        else:
            return "Heavy Metal"

    return "General"