"""
Virtual-line crossing detector.

Keeps the last known centroid y-position (for a horizontal line) for every
track id, and flags a "crossing event" the first time a track's centroid
moves from one side of the line to the other. Each track id can only ever
fire one crossing event, so a vehicle is never double-counted.
"""


class LineCounter:
    def __init__(self, line_y: int, orientation: str = "horizontal"):
        """
        line_y: pixel y-coordinate (or x-coordinate if orientation='vertical')
                of the virtual counting line.
        """
        self.line_y = line_y
        self.orientation = orientation
        self.last_pos = {}      # track_id -> last centroid coordinate
        self.counted_ids = set()  # track_ids that already triggered a crossing
        self.counts = {}        # vehicle_type -> count

    def _centroid_coord(self, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return cy if self.orientation == "horizontal" else cx

    def update(self, track_id: int, box, vehicle_type: str) -> bool:
        """
        Call once per frame per tracked object.
        Returns True exactly once per track_id, the frame its centroid
        crosses the line.
        """
        coord = self._centroid_coord(box)
        crossed = False

        if track_id in self.last_pos and track_id not in self.counted_ids:
            prev = self.last_pos[track_id]
            if prev < self.line_y <= coord or prev > self.line_y >= coord:
                crossed = True
                self.counted_ids.add(track_id)
                self.counts[vehicle_type] = self.counts.get(vehicle_type, 0) + 1

        self.last_pos[track_id] = coord
        return crossed

    def summary(self):
        return dict(self.counts)
