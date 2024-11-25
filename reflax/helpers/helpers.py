def polanalyze(polstate):
    if polstate.lower() == "linear te/perpendicular/s":
        return 1.0, 0.0
    elif polstate.lower() == "linear tm/parallel/p":
        return 0.0, 1.0
    else:
        raise ValueError("Unsupported polarization state")