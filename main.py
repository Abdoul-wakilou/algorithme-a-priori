"""
Nom: TIGA
Prenom: Abdoul-Wakilou
Filière: Génie Logiciel - IFRI de l'Université d'Abomey-Calavi
Email: abdoulwakiloutiga@gmail.com
"""
import streamlit as st
import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time

def count_patterns(data):
    pattern_counts = {}

    products = ["A", "B", "C", "D"]
    for r in range(1, len(products) + 1):
        for combo in combinations(products, r):
            pattern_counts[''.join(combo)] = 0

    for _, row in data.iterrows():
        for pattern in pattern_counts.keys():
            if set(pattern).issubset(set(row)):
                pattern_counts[pattern] += 1

    return pattern_counts

def preprocess_data(data):
    products = set()
    for row in data.values:
        products.update(set(row))

    binary_data = pd.DataFrame(columns=list(products))
    for product in products:
        binary_data[product] = data.apply(lambda row: 1 if product in row.values else 0, axis=1)

    return binary_data

def main():
    st.title("Algorithme d'à priori")

    data = pd.read_csv("dataset.csv", header=None)

    st.write("Contenu du fichier dataset.csv :")
    st.write(data)

    support_patterns = st.number_input("Veuillez entrer le support pour l'analyse des patterns fréquents :", min_value=1, step=1)

    pattern_counts = count_patterns(data)

    motifs = []
    occurrences = []

    for pattern, count in pattern_counts.items():
        if count >= support_patterns:
            motifs.append(pattern)
            occurrences.append(count)

    st.write(f"Les patterns les plus fréquents avec un support d'au moins {support_patterns} :")
    df_patterns = pd.DataFrame({'Pattern': motifs, 'Support du pattern': occurrences})
    st.table(df_patterns)

    binary_data = preprocess_data(data)

    support_association = st.number_input("Veuillez entrer un support pour les règles d'association :", min_value=1, step=1)

    frequent_itemsets = apriori(binary_data, min_support=support_association/binary_data.shape[0], use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values(by='lift', ascending=False).head(10)

    rules = rules.rename(columns={"antecedents": "Produits associés", "consequents": "Produits recommandés"})

    st.write(f"Top 10 des recommandations les plus bénéfiques avec un support d'au moins {support_association} :")
    st.table(rules)

    execution_times = []
    supports = [i for i in range(1, 101)]

    for support in supports:
        start_time = time.time()

        frequent_itemsets = apriori(binary_data, min_support=support/binary_data.shape[0], use_colnames=True)

        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)

    st.write("Évolution du temps d'exécution en fonction du support :")
    st.line_chart(pd.DataFrame({"Support (%)": supports, "Temps d'exécution (s)": execution_times}))

if __name__ == "__main__":
    main()
