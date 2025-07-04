# test_export.py
import pandas as pd
from export.export_results import export_results

# Dummy test data
df = pd.DataFrame([
    {
        "title": "Health economics of snakebite envenomation: A sub-Saharan African perspective.",
        "abstract": "Sub-Saharan Africa (SSA) is affected by the high direct and indirect costs of snakebite envenomation. With >30% of global mortality, different economic barriers still exist, and effective strategies must be employed to avert the burden and promote quality of life. With the WHO target of reducing the number of snakebites by one-half by 2030, different aspects concerning snakebite envenomation economics must be evaluated, and potential strategies must be developed. Strategies such as exploring the different snakebite prevention interventions, and the costs associated with these interventions, must be prioritized through extensive research and targeted surveys. Information obtained from these surveys can be used to draft effective policies to minimize snakebite envenomation incidence, reduce the economic burden associated with envenomation and improve the quality of life of people at risk. In this narrative review, we evaluate the different aspects concerning the health economics of snakebite envenomation and explore the financial capacity of SSA countries to mitigate envenomation. Additionally, we propose multiple steps that could be undertaken to mitigate the financial burden of envenomation in SSA. Furthermore, we propose critical research strategies to minimize direct and indirect costs arising from snakebite envenomation in the region.",
        "authors": "Ayesiga I, Gmanyami J, Akaka A, et al. ",
        "pub_year": 2025,
        "pub_month": '',
        "doi": "http://dx.doi.org/10.1093/trstmh/trae062",
        "source": "PubMed",
        "inclusion_prediction_bert": 1,
        "inclusion_probability_bert": 0.92,
        "inclusion_prediction_svm": 1,
        "inclusion_probability_svm": 0.98,
        "disease_predictions": ["Snakebite envenoming"]
    },
    
    {
        "title": "Vaccines against Chagas’ disease: a synthetized up-to-date review",
        "abstract": "Chagas’ disease (CD) is an infectious disease attacking an estimated 8 million people, mainly in rural areas of Latin America countries. CD has no effective treatment, evidencing the vaccination schedule as the best control strategy. Although some medicaments are available, none of them provides a solution for the infection nor are capable of inducing protection. They also have questionable safety levels and side effects. In light of this, several experimental vaccines are in development in order to improve safety, reproducibility, and protective immune response against the etiologic agent of CD, Trypanosoma cruzi. In this review, we discuss aspects as antigen, adjuvant, routes of administration, protection level, animal models, and economic impact in CD vaccine development, as well the challenges and future perspectives.",
        "authors": "Santos GSD, Fonseca BDR, Dall’Agno L",
        "pub_year": 2025,
        "pub_month": 3,
        "doi": "10.1016/j.medj.2022.05.004",
        "source": "CrossRef",
        "inclusion_prediction_bert": 1,
        "inclusion_probability_bert": 0.99,
        "inclusion_prediction_svm": 1,
        "inclusion_probability_svm": 0.99,
        "disease_predictions": ["Chagas disease"]
    },
    {
        "title": "One Health Networks for Infectious Diseases Surveillance and Pandemic Preparedness in Central and South America",
        "abstract": "The SARS-CoV-2 2019 pandemic prompted the emergence of collaborative initiatives within South America and the Caribbean, to tackle common challenges. Many initiatives included local government, international entities, military, academia, and research institutions, united to face the challenges brought by the pandemic. Some collaborations were new, but most were built on top of existing networks developed to prevent and control challenges like zoonotic diseases. In the last 40 years, the U.S. Naval Medical Research Unit (NAMRU) SOUTH has helped ensure the readiness and health of U.S. service members, Peruvian partners, and civilian population through research, surveillance, and global health, covering One Health interconnectedness of human, animal, and environmental health to address zoonotic diseases, antimicrobial resistance, and vector-borne diseases. This article puts together the different communications, data sharing, and initiatives developed throughout South America towards One Health surveillance, focusing on zoonotic pathogens, and to describe the best practices for these networks.",
        "authors": "Guezala MC, Schilling MA",
        "pub_year": 2025,
        "pub_month": 1,
        "doi": "http://dx.doi.org/10.1093/infdis/jiae571",
        "source": "CrossRef",
        "inclusion_prediction_bert": 1,
        "inclusion_probability_bert": 0.87,
        "inclusion_prediction_svm": 1,
        "inclusion_probability_svm": 0.94,
        "disease_predictions": ["Leprosy"]
    },
])

# Only include models actually used
used_models = ["bert", "svm"]

# Run export
export_results(df, used_models)

