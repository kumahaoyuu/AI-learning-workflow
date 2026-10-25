"""Hand-designed, fully observed discrete Bayesian network, using only NumPy.

Every feature has Y as a parent. FEATURE_PARENTS gives its optional additional
feature parent. Indices are zero based here; the report uses F1 through F12.
"""

import numpy as np


FEATURE_NAMES = (
    "top_left_gradient_axis", "top_center_gradient_axis", "top_right_gradient_axis",
    "middle_left_gradient_axis", "middle_center_gradient_axis", "middle_right_gradient_axis",
    "bottom_left_gradient_axis", "bottom_center_gradient_axis", "bottom_right_gradient_axis",
    "bounding_box_aspect_ratio", "upper_enclosed_background_fraction", "lower_enclosed_background_fraction",
)
FEATURE_PARENTS = (None, 0, 1, None, 3, 4, None, 6, 7, None, 9, 10)
N_STATES = 5
ORIENTATION_EDGES = (0.20, 0.40, 0.60, 0.80)
ASPECT_EDGES = (0.45, 0.60, 0.75, 0.90)
HOLE_EDGES = (0.001, 0.03, 0.10, 0.22)
REGION_BOUNDARIES = (0, 8, 16, 24)
FOREGROUND_THRESHOLD = 64
NORMALIZED_SIZE = 24
HOLE_NORMALIZED_SIZE = 21


def _validate_X(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError("X must have shape (n_samples, 784).")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError("X must contain real numeric pixels.")
    if not np.isfinite(X).all() or (X.size and (X.min() < 0 or X.max() > 255)):
        raise ValueError("X pixels must be finite and in [0, 255].")
    return X


def _validate_y(y, n):
    y = np.asarray(y)
    if y.shape != (n,) or n == 0:
        raise ValueError("y must have shape (n_samples,); training cannot be empty.")
    if not np.issubdtype(y.dtype, np.number) or np.iscomplexobj(y):
        raise ValueError("y must contain integer labels in [0, 9].")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 9) | (y != np.floor(y))):
        raise ValueError("y must contain integer labels in [0, 9].")
    return y.astype(np.int64, copy=False)


def _enclosed_background(binary):
    """Find background not 4-connected to the image boundary; no external library."""
    background = ~binary
    outside = np.zeros_like(background)
    outside[:, 0, :] = background[:, 0, :]
    outside[:, -1, :] = background[:, -1, :]
    outside[:, :, 0] = background[:, :, 0]
    outside[:, :, -1] = background[:, :, -1]
    while True:
        grown = outside.copy()
        grown[:, 1:, :] |= outside[:, :-1, :]
        grown[:, :-1, :] |= outside[:, 1:, :]
        grown[:, :, 1:] |= outside[:, :, :-1]
        grown[:, :, :-1] |= outside[:, :, 1:]
        grown &= background
        if np.array_equal(grown, outside):
            break
        outside = grown
    return background & ~outside


def _deskew_pixels(X):
    """Remove horizontal slant using gray-level second moments, without labels."""
    X = np.asarray(X)
    output = np.empty(X.shape, dtype=np.float32)
    xx = np.arange(28, dtype=np.float64)[None, None, :]
    yy = np.arange(28, dtype=np.float64)[None, :, None]
    for start in range(0, len(X), 1024):
        images = X[start:start + 1024].reshape(-1, 28, 28).astype(np.float64)
        mass = images.sum(axis=(1, 2))
        safe_mass = np.maximum(mass, 1)
        cx = (images * xx).sum(axis=(1, 2)) / safe_mass
        cy = (images * yy).sum(axis=(1, 2)) / safe_mass
        dx, dy = xx - cx[:, None, None], yy - cy[:, None, None]
        mu11 = (images * dx * dy).sum(axis=(1, 2))
        mu02 = (images * dy * dy).sum(axis=(1, 2))
        slope = np.divide(mu11, mu02, out=np.zeros_like(mu11), where=mu02 > 1e-12)
        slope = np.clip(slope, -1, 1)
        source_x = xx + slope[:, None, None] * dy
        left = np.floor(source_x).astype(np.int64)
        right = left + 1
        fraction = source_x - left
        a = np.take_along_axis(images, np.clip(left, 0, 27), axis=2)
        b = np.take_along_axis(images, np.clip(right, 0, 27), axis=2)
        result = a * (1 - fraction) * ((left >= 0) & (left < 28))
        result += b * fraction * ((right >= 0) & (right < 28))
        output[start:start + len(images)] = result.reshape(-1, 784)
    return output


def _resize_box(X,size=21,threshold=64,keep_aspect=False,canvas=None,center_mass=False):
    """Deterministic gray bilinear box sampling, optionally onto a 28x28 canvas."""
    output=np.empty((len(X),canvas or size,canvas or size),dtype=np.float32)
    for start in range(0,len(X),512):
        raw=X[start:start+512].reshape(-1,28,28).astype(np.float64)
        mask=raw>threshold
        rows,cols=mask.any(2),mask.any(1)
        nonempty=rows.any(1)
        top,left=rows.argmax(1),cols.argmax(1)
        height=28-top-rows[:,::-1].argmax(1)
        width=28-left-cols[:,::-1].argmax(1)
        oh=np.full(len(raw),size)
        ow=np.clip(np.rint(size*width/height),1,(canvas or size)-2).astype(int) if keep_aspect else oh
        side=canvas or size
        oy=(side-oh)//2 if canvas else np.zeros(len(raw),int)
        ox=(side-ow)//2 if canvas else np.zeros(len(raw),int)
        yy=np.arange(side)[None,:,None]
        xx=np.arange(side)[None,None,:]
        sy=top[:,None,None]+np.clip((yy-oy[:,None,None]+0.5)*height[:,None,None]/oh[:,None,None]-0.5,0,height[:,None,None]-1)
        sx=left[:,None,None]+np.clip((xx-ox[:,None,None]+0.5)*width[:,None,None]/ow[:,None,None]-0.5,0,width[:,None,None]-1)
        y0,x0=np.floor(sy).astype(int),np.floor(sx).astype(int)
        y1,x1=np.minimum(y0+1,27),np.minimum(x0+1,27)
        fy,fx=sy-y0,sx-x0
        ids=np.arange(len(raw))[:,None,None]
        sampled=(raw[ids,y0,x0]*(1-fx)*(1-fy)+raw[ids,y0,x1]*fx*(1-fy)
                +raw[ids,y1,x0]*(1-fx)*fy+raw[ids,y1,x1]*fx*fy)
        valid=(yy>=oy[:,None,None])&(yy<(oy+oh)[:,None,None])&(xx>=ox[:,None,None])&(xx<(ox+ow)[:,None,None])
        sampled*=valid&nonempty[:,None,None]
        if center_mass:
            mass=sampled.sum((1,2))
            denom=np.maximum(mass,1)
            cy=(sampled*yy).sum((1,2))/denom
            cx=(sampled*xx).sum((1,2))/denom
            sy=yy+cy[:,None,None]-(side-1)/2
            sx=xx+cx[:,None,None]-(side-1)/2
            y0,x0=np.floor(sy).astype(int),np.floor(sx).astype(int)
            y1,x1=y0+1,x0+1
            fy,fx=sy-y0,sx-x0
            recentered=np.zeros_like(sampled)
            for iy,wy in ((y0,1-fy),(y1,fy)):
                for ix,wx in ((x0,1-fx),(x1,fx)):
                    recentered+=sampled[ids,np.clip(iy,0,side-1),np.clip(ix,0,side-1)]*wy*wx*((iy>=0)&(iy<side)&(ix>=0)&(ix<side))
            sampled=recentered
        output[start:start+len(raw)]=sampled
    return output


def _gradient_histogram(images,rows,cols,n_angles=4,soft=True):
    """Fixed unsigned gradient orientation mass in explicitly located regions."""
    X=images.astype(np.float32)/255
    gx=np.zeros_like(X);gy=np.zeros_like(X)
    gx[:,:,1:-1]=(X[:,:,2:]-X[:,:,:-2])/2
    gy[:,1:-1,:]=(X[:,2:,:]-X[:,:-2,:])/2
    mag=np.sqrt(gx*gx+gy*gy)
    angle=np.mod(np.arctan2(gy,gx),np.pi)*n_angles/np.pi
    low=np.floor(angle).astype(np.int64)%n_angles
    fraction=angle-np.floor(angle)
    hist=[]
    rb=np.linspace(0,X.shape[1],rows+1,dtype=int)
    cb=np.linspace(0,X.shape[2],cols+1,dtype=int)
    for r in range(rows):
        for c in range(cols):
            sl=(slice(None),slice(rb[r],rb[r+1]),slice(cb[c],cb[c+1]))
            columns=[]
            for d in range(n_angles):
                weights=(low[sl]==d).astype(float)
                if soft:
                    weights=weights*(1-fraction[sl])+((low[sl]+1)%n_angles==d)*fraction[sl]
                columns.append((mag[sl]*weights).sum((1,2)))
            hist.append(np.stack(columns,axis=1))
    return np.stack(hist,axis=1)


def _gradient_axis_features(images):
    """Unsigned gradient main-axis angles divided by pi, one for each 8x8 cell.

    Compute four softly weighted unsigned orientation bins, then use their
    magnitude-weighted doubled-angle circular mean. Zero-gradient cells have
    angle zero by convention. This is a geometric image statistic, not a
    classifier or a learned projection.
    """
    hist = _gradient_histogram(images, 3, 3, 4, True)
    theta = np.arange(4) * np.pi / 4
    cosine = (hist * np.cos(2 * theta)).sum(axis=2)
    sine = (hist * np.sin(2 * theta)).sum(axis=2)
    return np.mod(np.arctan2(sine, cosine), 2 * np.pi) / (2 * np.pi)


def _continuous_features(X):
    """Nine regional gradient axes, deskewed-box aspect, upper/lower hole area.

    Both geometry branches use the same >64 box after per-image moment deskew.
    Directions use a bilinear 24x24 box image; hole masks are determined on the
    deskewed 28x28 image and sampled to 21x21 with nearest-neighbor coordinates.
    """
    X = _validate_X(X)
    features = np.empty((len(X), 12), dtype=np.float64)
    for start in range(0, len(X), 1024):
        pixels = _deskew_pixels(X[start:start + 1024])
        images = _resize_box(pixels, size=NORMALIZED_SIZE, threshold=FOREGROUND_THRESHOLD)
        block = features[start:start + len(pixels)]
        block[:, :9] = _gradient_axis_features(images)
        raw = pixels.reshape(-1, 28, 28)
        binary = raw > FOREGROUND_THRESHOLD
        rows, cols = binary.any(axis=2), binary.any(axis=1)
        nonempty = rows.any(axis=1)
        top, left = rows.argmax(axis=1), cols.argmax(axis=1)
        height = 28 - top - rows[:, ::-1].argmax(axis=1)
        width = 28 - left - cols[:, ::-1].argmax(axis=1)
        block[:, 9] = np.where(nonempty, width / height, 0.0)
        sample_r = top[:, None] + np.arange(HOLE_NORMALIZED_SIZE)[None, :] * height[:, None] // HOLE_NORMALIZED_SIZE
        sample_c = left[:, None] + np.arange(HOLE_NORMALIZED_SIZE)[None, :] * width[:, None] // HOLE_NORMALIZED_SIZE
        holes = _enclosed_background(binary)
        ids = np.arange(len(raw))[:, None, None]
        normalized_holes = holes[ids, sample_r[:, :, None], sample_c[:, None, :]]
        block[:, 10] = normalized_holes[:, :10, :].mean(axis=(1, 2))
        block[:, 11] = normalized_holes[:, 10:, :].mean(axis=(1, 2))
    return features


def extract_features(X):
    """Return fixed, batch-independent integer states, shape (n_samples,12).

    Each feature has five states. Ties go to the higher bin. No learned state,
    labels, quantiles or caller-side feature preprocessing are required.
    """
    continuous = _continuous_features(X)
    edges = [ORIENTATION_EDGES] * 9 + [ASPECT_EDGES, HOLE_EDGES, HOLE_EDGES]
    return np.column_stack([np.digitize(continuous[:, i], row) for i, row in enumerate(edges)])


class BayesNetClassifier:
    """Protocol v1. Y is the root; each feature has at most two total parents."""

    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive.")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = _validate_X(X)
        y = _validate_y(y, len(X))
        features = extract_features(X)
        self.classes_ = np.arange(10, dtype=np.int64)
        self.class_count_ = np.bincount(y, minlength=10)
        self.class_log_prior_ = np.log(
            (self.class_count_ + self.alpha) / (len(y) + 10 * self.alpha)
        )
        self.feature_parents_ = FEATURE_PARENTS
        self.cpts_ = []
        for index, parent in enumerate(self.feature_parents_):
            if parent is None:
                encoded = y * N_STATES + features[:, index]
                shape = (10, N_STATES)
            else:
                encoded = (y * N_STATES + features[:, parent]) * N_STATES + features[:, index]
                shape = (10, N_STATES, N_STATES)
            counts = np.bincount(encoded, minlength=int(np.prod(shape))).reshape(shape)
            # Normalize along the CHILD-state axis for every parent assignment.
            table = (counts + self.alpha) / (
                counts.sum(axis=-1, keepdims=True) + N_STATES * self.alpha
            )
            self.cpts_.append(table)
        self.log_cpts_ = [np.log(table) for table in self.cpts_]
        self.n_features_in_ = 784
        return self

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "class_log_prior_"):
            raise RuntimeError("Call fit before prediction.")
        features = extract_features(X)
        scores = np.broadcast_to(self.class_log_prior_, (len(features), 10)).copy()
        for index, parent in enumerate(self.feature_parents_):
            table = self.log_cpts_[index]
            if parent is None:
                scores += table[:, features[:, index]].T
            else:
                scores += table[:, features[:, parent], features[:, index]].T
        return scores

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
