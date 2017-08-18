import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def plot_tsne(t1,y_label,i):
    toy,toy_label=array_creator(t1,y_label)
    temp=model.fit_transform(toy,toy_label)
    name="layer"+str(i)+"retesting"
    scatter(temp, np.asarray(toy_label))
    plt.savefig(name, dpi=120)


def array_creator(toy,toy_label):
    test=zip(toy,toy_label)
    random.shuffle(test)
    return seaprator(test)

def seaprator(tups):
    toy=[]
    toy_label=[]
    for i in xrange(N):
        toy.append(tups[i][0])
        toy_label.append(tups[i][1])
    return toy, toy_label

from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)

