# % [python]


def get_classwise_documents(
    connection_url: str, database_name: str, classes: list[str]
) -> list[dict]:
    import pymongo

    # fetches class-wise vote documents from mongodb cluster and returns dictionary with values as array of vote documents
    conn = pymongo.MongoClient(connection_url)
    database = conn.get_database(database_name)
    all_documents: list[dict] = []
    vote_document: dict
    for class_name in classes:
        class_documents: list[dict] = []
        collection = database.get_collection(class_name)

        for vote_document in collection.find({}):
            class_documents.append(vote_document["vote_data"])  # type: ignore

        all_documents.append({"name": class_name, "votes": class_documents})

    return all_documents


def download_results():
    import json

    CONNECTION_URL = "mongodb+srv://vermadivij:elections@cluster1.kicphp2.mongodb.net/?retryWrites=true&w=majority&appName=cluster1"
    CLASSES = ['10A', '10B', '10C', '10D', '10E', '10F', '10G', '10H', '10I', '10J', '11A', '11B', '11C', '11D', '11E', '12A', '12B', '12C', '12D', '9A', '9B', '9C', '9D', '9E', '9F', '9G', '9H', '9I', '9J', 'absentees', 'candidates']  # fmt: off
    documents = get_classwise_documents(CONNECTION_URL, "votes", CLASSES)
    with open("votes.json", "w+") as file:
        file.write(json.dumps(documents))


# % [python]

# fmt: off
candidate_data = {
    "Captain Boy": [ "Aadityaraje Desai", "Abhichandra Charke", "Praneel Deshmukh", "Rachit Srivastava", ],
    "Captain Girl": [ "Tvisha Shah", "Gauravi Zade", "Kirthika Jayachander", "Naisha Rastogi", ],
    "Vice Captain Boy": [ "Kausar Chandra", "Sagnik Ghosh", "Avaneesh Mahalle", "Krishna Yadav", "Viren Jadhav", ],
    "Vice Captain Girl": [ "Ketaki Phalle", "Trisha Kandpal", "Riya Shirode", "Kavya Mehta", "Sumedha Vaidya", ],
}
# fmt: on
import json

import pandas as pd


def calculate_votes(votes_json: str):
    with open(votes_json) as file:
        classwise_votes: list[dict] = json.loads(file.read())

    votes_df = pd.DataFrame(
        [
            {
                "class": _class["name"],
                "candidate_name": candidate["name"],
                "post": candidate["post"],
            }
            for _class in classwise_votes
            for votes in _class["votes"]
            for candidate in votes
        ],
        columns=["class", "candidate_name", "post"],  # type: ignore
    )
    return votes_df


votes_df = calculate_votes("votes.json")

# % [markdown]
"""
# Postwise dataframes
"""

# % [python]

classwise_grouped = postwise_votes_df.groupby("post")

cb = votes_df.loc[classwise_grouped.groups["Captain Boy"]].drop("post", axis=1)
cg = votes_df.loc[classwise_grouped.groups["Captain Girl"]].drop("post", axis=1)
vcb = votes_df.loc[classwise_grouped.groups["Vice Captain Boy"]].drop("post", axis=1)
vcg = votes_df.loc[classwise_grouped.groups["Vice Captain Girl"]].drop("post", axis=1)

postwise_votes_df = {
    "captain_boy": cb,
    "captain_girl": cg,
    "vice_captain_boy": vcb,
    "vice_captain_girl": vcg,
}

# print(cb)
# print(cg)
# print(vcb)
# print(vcg)

# % [python]

from Query import SqliteDatabase

with SqliteDatabase("example.db") as query:
    for post_name, post_df in postwise_votes_df.items():
        query(f"drop table if exists {post_name};", is_updation=True)
        query(
            f"create table {post_name} (class varchar(255), candidate_name varchar(255));",
            is_updation=True,
        )

        rows = []
        for idx, (class_name, candidate_name) in post_df.iterrows():
            rows.append(f"('{class_name}', '{candidate_name}')")

        query(f"insert into {post_name} values" + ",".join(rows), is_updation=True)
        query(f"select * from {post_name} limit 10", table_heading=post_name)


################################

# candidate popularity trends - comparing candidate performances across classes

import matplotlib.pyplot as plt
import pandas as pd
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
                label=candidate_name.replace("_", " "),  # type: ignore
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
