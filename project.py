#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">About the Project</h1>\n<div class="text-box bulleted">\n    <h2 class="header">Overview</h2>\n    <p class="text">The project involves the statistical analysis of the official results of the Intra-School Council Elections using libraries like Pandas, Matplotlib & Seaborn along with SQL.</p>\n    <p class="text">The student council elections are held every year to choose representatives for different leadership roles in the school, managed entirely by student-led tech teams.</p>\n    <h2 class="header">Election Software</h2>\n    <p class="text">The (1)<a href="http://github.com/d1vij/electionsoftware" target="_blank" rel="noopener noreferrer"><em>Election Software</em></a> through which the elections were held is a fullstack application that I designed and developed, featuring a frontend built with HTML, CSS, and TypeScript, and a REST api by FastAPI (Python). Votes were securely stored in a MongoDB database.</p>\n    <h2 class="header">Report Generation</h2>\n    <p class="text">This report is in fact a single Jupyter notebook exported via a custom script to HTML and styled via CSS, (and then printed). Source of which is available at (2)<a href="http://github.com/d1vij/ip-proj" target="_blank" rel="noopener noreferrer"><em>GitHub Repo</em></a>.</p>\n</div>\n\n<div class="text-box">\n    <h2 class="header">References</h2>\n    <p class="text">(1) https://github.com/d1vij/electionsoftware</p>\n    <p class="text">(2) https://github.com/d1vij/ip-proj</p>\n</div>\n\n<div class="text-box">\n    <h2 class="header">Libraries used</h2>\n    <ol><li><em>Pymongo</em> -> for querying vote documents from MongoDB server</li><li><em>sqlite3</em> -> for querying local sql database</li><li><em>pandas</em> -> Fora data manipulation and analysis</li><li><em>seaborn</em> -> Graphing</li><li><em>matplotlib</em> -> Graphing</li></ol>\n</div>\n')


# ### Importing Stuff

# In[ ]:


import pymongo
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h3 class="header">Defining constants</h3>\n')


# In[ ]:


sns.set_theme()
def replace_spaces(string: str, replace_with=_UNDERSCORE):
    return string.replace(_SPACE, replace_with)


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Structure of Data</h1>\n<div class="text-box bulleted">\n    <p class="text">The Votes were stored indivisually in a <em>MongoDB</em> server as in the following structure</p>\n    <pre class="language-js">\n        <code class="language-js">\n<span class="comment">//example single vote document</span>\n{\n    "_id": {\n        "$oid": "68a1819ceff178ec25b66fbb" <span class="comment">           // internal mongodb document id</span>\n    },\n    "token": "b489737f-7997-430c-950f-b8c1b22f68c3", <span class="comment"> // A unique uuid4 based token identifing the vote session</span>\n    "client": "29", <span class="comment">                                  // Computer on which the vote was done</span>\n    "vote_data": [ <span class="comment">                                   // candidates voted by the voter</span>\n        {\n            "name": "Abhichandra Charke", \n            "post": "Captain Boy"\n        },\n        {\n            "name": "Gauravi Zade",\n            "post": "Captain Girl"\n        },\n        {\n            "name": "Kausar Chandra",\n            "post": "Vice Captain Boy"\n        },\n        {\n            "name": "Ketaki Phalle",\n            "post": "Vice Captain Girl"\n        }\n    ]\n}\n</code></pre>\n</div>\n<div class="text-box bulleted">\n    <p class="text"><em class="underlined">get_classwise_documents</em> -> The function returns array of <em>T_Class_Documents</em> objects</p>\n    <p class="text">In which each object contains properties<ol><li><em class="underlined">name</em> -> The name of class </li><li><em class="underlined">votes</em> -> Array of <em>T_Vote</em></li></ol></p>\n    <p class="text">T_Vote contains two properties<ol><li><em class="underlined">name</em> -> Name of candidate voted</li><li><em class="underlined">post</em>-> Post the candidate is voted for</li></ol></p>\n</div>\n')


# In[ ]:



# Post-wise canddiate names
# empty dictionary to store totaled votes data
empty_votes_dict = {
    post_name: {
        class_name: {name: 0 for name in candidate_data[post_name]}
        for class_name in CLASSES
    }
    for post_name in candidate_data.keys()
}


def calculate_total_votes_of_class(
    class_documents: T_Class_Documents,
) -> T_Classwise_Postwise_Totals:

    compiled_votes = {
        post_name: {candidate_name: 0 for candidate_name in candidate_data[post_name]}
        for post_name in candidate_data.keys()
    }

    vote_sesh: List[T_Vote]

    for vote_sesh in class_documents["votes"]: #type: ignore
        for vote_document in vote_sesh:
            post = vote_document["post"]
            name = vote_document["name"]
            compiled_votes[post][name] += 1

    return {"name": class_documents["name"], "votes": compiled_votes}


def create_dataframes(compiled_votes: list[T_Classwise_Postwise_Totals]):

    # creating empty dataframes
    dataframes = {
        post_name: pd.DataFrame(0, CLASSES, candidates)
        for post_name, candidates in candidate_data.items()
    }


    for _class in compiled_votes:
        class_name = _class["name"]

        for post_name, votes in _class["votes"].items():
            dataframes[post_name].loc[class_name] = votes # type: ignore
    return dataframes


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<div class="text-box">\n  <p class="text">\n    The <em>query()</em> function is a wrapper around SQLite queries, supporting both \n    data retrieval and data modification. \n  </p>\n</div>\n')


# In[ ]:


import sqlite3
from sqlite3 import Connection, Cursor, OperationalError
from typing import Any, Literal


class SqliteDatabase:
    def __init__(self, database: str):
        self.database = database
        self.conn: Connection | None = None
        self.cursor: Cursor | None = None

    def __enter__(self):
        try:
            self.conn = sqlite3.connect(self.database)
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"Error occured in connecting to the database {self.database}. Error Details: {e}")

        return self.query

    def __exit__(self, exc_type, exc, tb):
        assert self.conn is not None
        assert self.cursor is not None

        self.cursor.close()
        self.conn.close()

        return False  # dont suppress the error

    def query(
        self,
        query: str,
        *,
        is_updation=False,  # is the current query contains some kind of updation ?? Doesnt return anything if true
        return_rows: None | Literal["str"] | Literal["tuple"] = None,
        table_heading: str | None = None,  # Title printed before printing output
    ) -> None | tuple[tuple[str, ...], ...]:
        assert self.conn is not None
        assert self.cursor is not None

        try:
            results = self.cursor.execute(query)
            self.conn.commit()
        except Exception as err:
            print("** Row / Column names with spaces should be enlcosed within quotes **")
            raise err

        if is_updation:
            return

        rows: list[Any] = results.fetchall()
        columns_headers: tuple[str, ...] = tuple(str(col[0]) for col in results.description)

        lines: tuple[tuple[str, ...], ...] = tuple((columns_headers, *rows))

        if return_rows == "tuple":
            return lines
        elif return_rows is None:
            # printing table header if provided
            if table_heading is not None:
                print(table_heading)

            # Finding max column width
            column_widths: list[int] = []

            for col_idx in range(len(lines[0])):
                widths = []
                for row_idx in range(len(lines)):
                    widths.append(len(str(lines[row_idx][col_idx])))
                column_widths.append(max(widths))

            # Printing column headers
            border_top_bottom = "+" + "-" * (sum(column_widths) + 3 * len(column_widths) - 1)  + "+"
            print(border_top_bottom)
            print("| ", end="")
            for idx, col_label in enumerate(lines[0]):
                print(str(col_label).ljust(column_widths[idx]), end=" | ")
            print()
            print(border_top_bottom)

            for row in rows[1:]:
                print("| ", end="")
                for idx, col_val in enumerate(row):
                    print(str(col_val).ljust(column_widths[idx]), end=" | ")
                print()

            print(border_top_bottom)
            return None


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Creating post dataframes and saving them to SQLite database</h1>\n')


# In[ ]:


class_wise_documents = get_classwise_documents()
result_dataframes = create_dataframes(
    list(map(calculate_total_votes_of_class, class_wise_documents))
)

print("Found Posts", *result_dataframes.keys(), sep="\n")

conn = sqlite3.connect(DATABASE_NAME + ".db")
cursor = conn.cursor()
for name, post_df in result_dataframes.items():
    name = name.replace(_SPACE, _UNDERSCORE)
    post_df.columns = [name.replace(_SPACE, _UNDERSCORE) for name in post_df.columns]
    post_df.to_sql(name, conn, if_exists="replace", index_label="Class")
    conn.commit()
conn.close()


# In[ ]:


# captian boy dataframe
cb = result_dataframes["Captain Boy"]
# captian girl dataframe
cg = result_dataframes["Captain Girl"]
# vice captian boy dataframe
vcb = result_dataframes["Vice Captain Boy"]
# vice captian girl dataframe
vcg = result_dataframes["Vice Captain Girl"]


# In[ ]:


print(f"Total voters : {cb.sum().sum()}")
print(f"Total classes: {len(CLASSES)}")


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '\n<h1 class="header">Statistical analysis</h1>\n<div class="text-box">\n\n</div>\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Total votes across all classes</h1>\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Total votes recieved by any Candidate</h1>\n<div class="text-box">\n    <p class="text">Working:</p><ol><li><p class="text">Defines <em>positions</em> for mapping each post to a subplot.</p></li><li><p class="text">Iterating over post dataframes in <em>result_dataframes</em>.</p></li><li><p class="text">Running SQL query to sum votes of in the table of that post via <em>query()</em>.</p></li><li><p class="text">Plotting a Seaborn bar chart of the total votes (<em>post_df.sum()</em>) in its respective subplot.</p></li></ol>\n</div>\n')


# In[ ]:


fig, axes = plt.subplots(2,2, figsize=(15,10))
positions = [(0,0), (0,1), (1,0), (1,1)]

for idx, (post_name, post_df) in enumerate(result_dataframes.items()):
    query(
        f"""
        select {', '.join([f"sum({name}) as {name}" for name in post_df.columns])}
        from {replace_spaces(post_name)}
        """,
        table_heading="Total Votes for - " + post_name
    )
    print()
    sns.barplot(post_df.sum(), ax=axes[positions[idx]]) #type: ignore
    axes[positions[idx]].set_title(post_name)
plt.tight_layout()
plt.show()
print()


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Candidate popularity trends</h1>\n<div class="text-box">\n    <p class="text">Comparing candidate performances across classes</p>\n    <p class="text">Following steps taken for each post\'s dataframe</p>\n    <div class="text-box bulleted">\n        <p class="text">Iterating over post dataframes and extracting all the rows belonging to a particular \'standard\' from the post\'s dataframe by using <em>Regular Expressions</em></p>\n        <p class="text">Dividing plot into 4 subplots for each class <em>(9, 10, 11, 12)</em></p>\n        <p class="text">Plotting the section-wise votes recieved by a candidate</p>\n    </div>\n</div>\n')


# In[ ]:


# candidate popularity trends - comparing candidate performances across classes

from matplotlib.ticker import MultipleLocator


def plot_popularity_trends(post_name: str, post_df: pd.DataFrame):

    # extracting rows belonging to a particular class from the post's dataframe using regular expressions
    class_wise_dataframes = [
        post_df[post_df.index.str.contains(_re)]
        for _re in [r"9\w", r"10\w", r"11\w", r"12\w"]  # <--- regex btw
    ]

    # dividing the plot into 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))

    subplot_positions = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]  # since there are only 4 classes / subplots
    linestyles = [":", "-", "--", "-.", "solid"]

    for idx in range(4):
        pos = subplot_positions[idx]
        class_df = class_wise_dataframes[idx]
        sections = class_df.index

        for idx, (candidate_name, candidate_series) in enumerate(class_df.items()):
            # plotting a subplot for each class
            axes[pos].plot(
                sections,
                candidate_series,
                label=candidate_name.replace(_UNDERSCORE, _SPACE), #type: ignore
                linestyle=linestyles[idx],
            )

        axes[pos].set_xlabel("class")
        axes[pos].set_ylabel("Votes")

        # axes[pos].set_ylim(0, post_df.max().max() + 1)

        # values on y-axis would have a difference of 2
        axes[pos].yaxis.set_major_locator(MultipleLocator(2))

    fig.suptitle(post_name, fontsize=32)

    # setting a common legend for the whole plot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, fontsize=15)

    plt.show()


plot_popularity_trends("Captain Boy", cb)
plot_popularity_trends("Captain Girl", cg)
plot_popularity_trends("Vice Captain Boy", vcb)
plot_popularity_trends("Vice Captain Girl", vcg)


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Plotting the share in percent of classes in which a candidate has majority</h1>\n<div class="text-box">\n    <p class="text">Working:</p>\n    <ol><li><p class="text">Dividing the plot into four subplots</p></li><li><p class="text">Extracting the count of classes in which a particular candidate has the maximum votes amongst all other candidates of same post</p></li><li><p class="text">Dividing the series obtained in previous step witht the total number of votes to get the percent share series</p></li><li><p class="text">Plotting the percent share series</p></li></ol>\n</div>\n')


# In[ ]:


total_classes = len(cb.index)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
fig.suptitle("Share (percent) of Classes in which a Candidate has a majority    ", fontsize=20)

subplot_positions = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

colors = plt.cm.copper_r(np.linspace(0,0.50,5)) # type: ignore


for idx, (post_name, post_df) in enumerate(result_dataframes.items()):
    pos = subplot_positions[idx]

    # Column wise maximum will give the winning candidate of that class
    classes_won_by_candidate_series = post_df.idxmax(1)
    count_series = (classes_won_by_candidate_series
                        .groupby(classes_won_by_candidate_series)
                        .count()
                        .sort_values(ascending=False)
                        )
    percents_series = count_series / total_classes

    max_val = percents_series.max()

    axes[pos].set_title(post_name)
    axes[pos].pie(
        percents_series,
        labels=percents_series.index.map(
            lambda name: name.replace(_UNDERSCORE, _SPACE)
        ),
        autopct="%1.1f%%",
        startangle=180,
        colors=colors,
    )
    axes[pos].set(aspect='equal')


plt.show()


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<div class="text-box bulleted">\n    <p class="text">Furthermore we can notice despite being the second leading candidate (Aadityaraje Desai), they have the almost double the class-wise majority share than the leading candidate (Praneel Deshmukh) for the post of School Captain</p>\n    <p class="text">The absence of fifth candidate (Riya Shirode) in the fourth pie shows that they are not the majority in any class amongst all other candidates of the same post</p>\n</div>\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Candidate Co-Voting Patterns</h1>\n\n<div class="text-box bulleted">\n    <p class="text">Analyzes whether voters who supported one candidate also tended to support another.</p>\n    <p class="text">Working:</p>\n</div>\n<div class="text-box">\n    <ol>\n        <li><p class="text">Builds a <em>vote matrix</em> recording which candidates were chosen in each voting session.</p><pre class="language-py"><code class="language-py"><p class="text comment">// Example</p>\n        Captain Boy     Captain Girl  Vice Captain Boy Vice Captain Girl class\n25    Praneel Deshmukh  Gauravi Zade      Viren Jadhav       Kavya Mehta   10C\n26   Aadityaraje Desai  Gauravi Zade  Avaneesh Mahalle       Kavya Mehta   10C\n27  Abhichandra Charke  Gauravi Zade  Avaneesh Mahalle     Ketaki Phalle   10C\n28    Praneel Deshmukh  Gauravi Zade      Sagnik Ghosh    Sumedha Vaidya   10C\n29  Abhichandra Charke  Gauravi Zade      Sagnik Ghosh    Sumedha Vaidya   10C\n30  Abhichandra Charke  Gauravi Zade  Avaneesh Mahalle     Ketaki Phalle   10C</code></pre></li>\n<li><p class="text">Constructs a <em>co-occurrence matrix</em> showing how often Candidate B was voted when Candidate A was voted.<pre class="language-py"><code class="language-py"><p class="comment">// Example</p>\n                    Gauravi Zade  Kirthika Jayachander  Naisha Rastogi\nAvaneesh Mahalle            94                     4              26   \nKrishna Yadav               54                     3              14   \nViren Jadhav                33                     2              11   \nKetaki Phalle              116                     5              38   \nTrisha Kandpal              52                     5              15   \nRiya Shirode                 5                     3               0   \nKavya Mehta                 52                     3              18   \nSumedha Vaidya              27                     4               1   \n</code></pre></p></li><li><p class="text">Normalizes it into a <em>probability matrix</em> to estimate the likelihood of co-support between candidates. Each row of the co-occurrence matrix is divided by the total votes in that row. This converts raw counts into conditional probabilities, i.e., the chance of Candidate B being voted given that Candidate A was voted.</p></li><li><p class="text">Visualizes both matrices using heatmaps — one for raw counts, the other for probabilities.</p></li></ol>\n</div>\n')


# In[ ]:


[*candidate_data.keys(), "class"]


# In[ ]:


# constructing votes matrix
# vote matrix contains which candidate was voted for which post in any particular voting session
vote_matrix = pd.DataFrame(columns=[*candidate_data.keys(), "class"])
for _class in class_wise_documents:
    for session_votes in _class["votes"]:

        idx = len(vote_matrix)
        vote_dict = {}
        vote_dict["class"] = _class["name"]
        for vote in session_votes:
            vote_dict[vote["post"]] =  vote["name"]  # type:ignore
        vote_matrix.loc[idx] = vote_dict

all_candidates = []
for _, candidates in candidate_data.items():
    all_candidates.extend(candidates)
print(vote_matrix.iloc[25:31])


# In[ ]:


vote_only_matrix = vote_matrix[["Captain Boy", "Captain Girl", "Vice Captain Boy", "Vice Captain Girl"]]

# co-occurance matrix is the matrix showing how many times candidate B was voted when candidate A was voted
# co-occurance matrix would be N * N where N are the total number of candidates across all posts (18 * 18 for this case)
co_occurance_matrix = pd.DataFrame(0, index=all_candidates, columns=all_candidates)

# updating co-occurance matrix
for idx, session in vote_only_matrix.iterrows():
    for name_A in session.values:
        for name_B in session.values:
            if name_A != name_B:
                co_occurance_matrix.loc[name_A, name_B] += 1

# creating conditional probability matrix
# conditional probabilty matrix is created by normalizing columns of co-occurance matrix
# normalizing means dividing each row of co-occurance matrix by the total votes in that row
# the matrix gives the probabilty of person b (x axis) being voted when person A (y axis) was voted
probability_matrix = co_occurance_matrix.div(co_occurance_matrix.sum(axis=1), axis=0)

# ---- first plot ----
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title(
    "Co-occurance plot - Number of times person A got voted when person B was voted",
    size=16,
)
sns.heatmap(
    co_occurance_matrix,
    cmap="viridis",
    vmin=0,
    annot=True,
    ax=ax1,
    fmt=".0f",
    cbar_kws={"label": "Count"},
)
ax1.set_ylabel("Person A", size=12)
ax1.set_xlabel("Person B", size=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
plt.show()

# ---- second plot ----
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title(
    "Probability plot - Probabilty (in percent) of person B getting voted when person A was voted",
    size=16,
)
sns.heatmap(
    probability_matrix * 100,
    cmap="viridis",
    vmin=0,
    annot=True,
    ax=ax2,
    fmt=".1f",
    cbar_kws={"label": "Percent"},
)
ax2.set_ylabel("Person A", size=12)
ax2.set_xlabel("Person B", size=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
plt.show()

# the percents here for a column dont add up to 100 cuz they are mutually exclusive events


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<h1 class="header">Strongest & Weakest Allies</h1>\n\n<div class="text-box bulleted">\n    <p class="text">Identifies which candidates tend to appear most often (or least often) alongside another candidate in votes.</p>\n</div>\n\n<div class="text-box">\n    <p class="text">Working:</p>\n    <ol><li><p class="text">For each candidate, extracts their row from the <em>probability matrix</em> (probability of other candidates being voted when this candidate is chosen).</p></li><li><p class="text">Removes same-post candidates to avoid trivial overlaps (since voters pick only one per post).</p></li><li><p class="text">Finds the <strong>Strongest Ally</strong> → candidate with the highest co-vote probability.</p></li><li><p class="text">Finds the <strong>Weakest Ally</strong> → candidate with the lowest co-vote probability.</p></li><li><p class="text">Combines results into a summary table, showing strongest and weakest allies for each candidate.</p></li></ol>\n</div>\n')


# In[ ]:


# Use row indexes for all comparisions

strongest_ally_series = pd.Series(name="Strongest Ally")
weakest_ally_series = pd.Series(name="Weakest Ally")

for post_name, same_post_candidates in candidate_data.items():
    for name in same_post_candidates:
        # extracting the row which gives co-occurance probabilty for a candidate
        cps = probability_matrix.loc[name]

        # removing values of all the candidates in the same post
        candidate_probability_series = cps[~cps.index.isin(same_post_candidates)]

        strongest_ally_series[name] = candidate_probability_series.idxmax()
        weakest_ally_series[name] = candidate_probability_series.idxmin()

# concat based on similar rows
summary = pd.concat([strongest_ally_series, weakest_ally_series], axis=1)
print(summary.sort_values(by=list(summary.columns)))

# strongest ally is the candidate who is most likely to be voted when a candidate is voted
# weakest ally is the candidate who is least likely to be voted when a candidate is voted


# In[ ]:


# mean co-support - mean of all conditional probabilties across all candidates
probability_matrix.mean() * 100


# In[ ]:


# exporting votes to csv
vote_only_matrix.to_csv("votes.csv")


# In[ ]:


unqiue_votes = vote_only_matrix.value_counts()
print(f"Most popular choice groups\n\n{unqiue_votes.head(3)}")


# In[207]:


a = vote_matrix.value_counts()
a.head()


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '<!-- TODO: Force this onto new page -->\n<h1 class="header">Raw csv votes</h1>\n')
