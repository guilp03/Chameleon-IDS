"""Ponto de entrada principal do Chameleon-IDS (dataset NSL-KDD).

Pipeline:
1. Pré-processamento do NSL-KDD
2. PSO para seleção de features e hiperparâmetros do ensemble
3. Busca aleatória de hiperparâmetros do Autoencoder
4. Treinamento e avaliação final do Autoencoder no conjunto de teste
"""
import logging
import random
import time

import pandas as pd
import torch
from joblib import Parallel, delayed

import chameleon.data.preprocessing as data
import chameleon.pso.optimizer as pso
import chameleon.pso.gwo_optimizer as gwo
from chameleon.config import ModelBounds, PSOConfig
from chameleon.models import autoencoder as ae
from chameleon.pso.particle import Particle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # ------------------------------------------------------------------
    # Configuração
    # ------------------------------------------------------------------
    ensemble_type = "gb"  # "gb" = GradientBoosting, "rf" = RandomForest
    optimizer_type = "pso"  # "pso" = clássico, "gwo" = PSO híbrido com GWO
    cfg = PSOConfig()

    # ------------------------------------------------------------------
    # Carregamento e pré-processamento do dataset
    # ------------------------------------------------------------------
    csv_path = "csv_result-KDDTrain+_20Percent.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset não encontrado: '{csv_path}'. "
            "Baixe o NSL-KDD em https://www.unb.ca/cic/datasets/nsl.html "
            "e coloque o arquivo no mesmo diretório que main.py."
        )

    df.columns = df.columns.str.replace("'", "")
    df = df.drop(labels="id", axis=1)
    df, column_names, y = data.preprocess_nslkdd(df)
    n_features = len(column_names)
    logger.info("Dataset carregado. Features: %d, Amostras: %d", n_features, len(df))
    logger.info("Optimizer: %s", optimizer_type.upper())

    # ------------------------------------------------------------------
    # Inicialização do enxame PSO
    # ------------------------------------------------------------------
    globalbest: list = []
    globalbest_val: float = 0.0
    globalbest_feat_number: int = 0

    start_time = time.time()

    def process_particle(i: int) -> Particle:
        initial_position = pso.search_space(ensemble_type, n_features=n_features)
        particle = Particle(i, initial_position, ensemble_type=ensemble_type, column_names=column_names)
        particle.pb_val = pso.evaluate_fitness(ensemble_type, particle, column_names, df, y, particle.index, n_features=n_features)
        particle.pos_val = particle.pb_val
        particle.pb_feat_number = len(data.particle_choices(particle.position, column_names, n_features=n_features))
        return particle

    swarm: list[Particle] = list(
        Parallel(n_jobs=-1)(delayed(process_particle)(i) for i in range(cfg.swarm_size))
    )

    # ------------------------------------------------------------------
    # Determinação do globalbest inicial
    # ------------------------------------------------------------------
    globalbest = swarm[0].position
    globalbest_val = swarm[0].pb_val
    globalbest_feat_number = swarm[0].pb_feat_number

    def find_globalbest(
        globalbest: list,
        globalbest_val: float,
        globalbest_feat_number: int,
        swarm: list[Particle],
    ) -> tuple[list, float, int]:
        for particle in swarm[1:]:
            if particle.pb_val > globalbest_val:
                globalbest_val = particle.pb_val
                globalbest = particle.position
                globalbest_feat_number = particle.pb_feat_number
            elif particle.pb_val == globalbest_val:
                if particle.pb_feat_number < globalbest_feat_number:
                    globalbest_val = particle.pb_val
                    globalbest = particle.position
                    globalbest_feat_number = particle.pb_feat_number
        return globalbest, globalbest_val, globalbest_feat_number

    globalbest, globalbest_val, globalbest_feat_number = find_globalbest(
        globalbest, globalbest_val, globalbest_feat_number, swarm
    )
    logger.info("globalbest inicial: val=%.4f, features=%d", globalbest_val, globalbest_feat_number)

    # ------------------------------------------------------------------
    # Iterações do PSO
    # ------------------------------------------------------------------
    def apply_pso(
        particle: Particle,
        X_alpha: Particle | None = None,
        X_beta: Particle | None = None,
        X_delta: Particle | None = None,
        c: float = 0.0,
        curr_iter: int = 0,
    ) -> Particle:
        if optimizer_type == "gwo":
            particle.velocity = gwo.check_velocity(
                globalbest=globalbest,
                particle=particle,
                inertia=cfg.inertia,
                c=c,
                X_alpha=X_alpha,
                X_beta=X_beta,
                X_delta=X_delta,
                curr_iteration=curr_iter,
                num_iterations=cfg.max_iterations,
            )
        else:
            particle.velocity = pso.check_velocity(
                globalbest=globalbest,
                particle=particle,
                inertia=cfg.inertia,
                c1=cfg.cognitive_param,
                c2=cfg.social_param,
            )
        particle.position = pso.update_particle(particle, ensemble_type, n_features=n_features)
        particle.pos_val = pso.evaluate_fitness(
            ensemble_type, particle, column_names, df, y, particle.index, n_features=n_features
        )
        particle = pso.update_pb(particle)
        particle.pb_feat_number = len(
            data.particle_choices(particle.position, column_names, n_features=n_features)
        )
        return particle

    if optimizer_type == "gwo":
        # Loop GWO com líderes alpha/beta/delta e contador m
        m = 0
        for iteration in range(cfg.max_iterations):
            logger.info("Iteração GWO: %d/%d", iteration + 1, cfg.max_iterations)
            X_alpha, X_beta, X_delta = gwo.find_leaders(
                swarm, globalbest_val, globalbest_feat_number
            )
            c = (m / cfg.max_iterations) ** (2 / 3) + 1
            curr_globalbest = globalbest_val
            swarm = list(
                Parallel(n_jobs=-1)(
                    delayed(apply_pso)(p, X_alpha, X_beta, X_delta, c, iteration)
                    for p in swarm
                )
            )
            globalbest, globalbest_val, globalbest_feat_number = find_globalbest(
                globalbest, globalbest_val, globalbest_feat_number, swarm
            )
            logger.info("globalbest: val=%.4f, features=%d", globalbest_val, globalbest_feat_number)
            if globalbest_val == curr_globalbest:
                m = 0
            else:
                m += 1
    else:
        for iteration in range(cfg.max_iterations):
            logger.info("Iteração PSO: %d/%d", iteration + 1, cfg.max_iterations)
            swarm = list(Parallel(n_jobs=-1)(delayed(apply_pso)(p) for p in swarm))
            globalbest, globalbest_val, globalbest_feat_number = find_globalbest(
                globalbest, globalbest_val, globalbest_feat_number, swarm
            )
            logger.info("globalbest: val=%.4f, features=%d", globalbest_val, globalbest_feat_number)

    # ------------------------------------------------------------------
    # Extração da solução ótima
    # ------------------------------------------------------------------
    optimal_solution = globalbest
    optimal_x_train, optimal_x_val, optimal_x_test, optimal_y_train, optimal_y_val, optimal_y_test = (
        data.get_optimal_subsets(df, optimal_solution, column_names, y, optimal_solution[0], n_features=n_features)
    )

    end_time = time.time()
    data.get_time(start_time, end_time)

    chosen_features = data.particle_choices(globalbest, column_names, n_features=n_features)
    logger.info("Features escolhidas (%d): %s", len(chosen_features), chosen_features)

    swarm[0].position = globalbest

    # ------------------------------------------------------------------
    # Preparação dos dados para o Autoencoder (apenas classe benigna)
    # ------------------------------------------------------------------
    optimal_x_train["class"] = optimal_y_train
    optimal_x_train = optimal_x_train.query("`class` == 0").drop(labels="class", axis=1)

    optimal_x_val["class"] = optimal_y_val
    benign_x_val_optimal = optimal_x_val[optimal_x_val["class"] == 0].drop(labels="class", axis=1)
    optimal_x_val = optimal_x_val.drop(labels="class", axis=1)

    optimal_x_train, optimal_x_val, optimal_x_test, benign_x_val_optimal = (
        data.transform_min_max_scaler(optimal_x_train, optimal_x_val, optimal_x_test, benign_x_val_optimal)
    )

    benign_x_val_tensor = torch.FloatTensor(benign_x_val_optimal)
    optimal_x_train_tensor = torch.FloatTensor(optimal_x_train)

    # ------------------------------------------------------------------
    # Busca aleatória de hiperparâmetros do Autoencoder
    # ------------------------------------------------------------------
    start_time = time.time()

    hyperparameter_ranges = {
        "BATCH_SIZE": [16, 32, 64, 128],
        "ALPHA": [1e-4, 1e-3, 1e-2, 9.6e-2, 1e-1],
        "PATIENCE": [10],
        "DELTA": [0.0001],
        "NUM_EPOCHS": [1000],
        "DROPOUT_RATE": [0.5],
        "REGULARIZER": [1e-4, 1e-3, 1e-2, 1e-1],
    }
    num_iterations = 80
    best_hyperparameters: dict = {}
    best_f1_score: float = float("-inf")
    best_threshold: float = 0.0
    best_precision: float = 0.0
    best_accuracy: float = 0.0
    best_tpr: float = 0.0
    best_fpr: float = 0.0
    best_recall: float = 0.0

    in_features = optimal_x_train.shape[1]

    for m in range(num_iterations):
        logger.info("Iteração Autoencoder: %d/%d", m + 1, num_iterations)
        hyperparameters = {param: random.choice(values) for param, values in hyperparameter_ranges.items()}

        ae_model = ae.Autoencoder(in_features, hyperparameters["DROPOUT_RATE"])
        ae_model.compile(
            learning_rate=hyperparameters["ALPHA"],
            weight_decay=hyperparameters["REGULARIZER"],
        )
        ae_model.fit(
            optimal_x_train_tensor,
            hyperparameters["NUM_EPOCHS"],
            hyperparameters["BATCH_SIZE"],
            X_val=benign_x_val_tensor,
            patience=hyperparameters["PATIENCE"],
            delta=hyperparameters["DELTA"],
        )

        val_anomaly_scores = ae.get_autoencoder_anomaly_scores(ae_model, optimal_x_val)

        thresholds = [x / 1000 for x in range(1, 5000)]
        best_f1 = 0.0
        best_thresh = 0.0
        current_precision = 0.0
        current_accuracy = 0.0
        current_tpr = 0.0
        current_fpr = 0.0
        current_recall = 0.0

        for thresh in thresholds:
            metrics = ae.get_overall_metrics(optimal_y_val, val_anomaly_scores > thresh)
            if metrics["f1-score"] > best_f1 and metrics["tpr"] > 0.0:
                best_f1 = metrics["f1-score"]
                best_thresh = thresh
                current_precision = metrics["precision"]
                current_accuracy = metrics["accuracy"]
                current_tpr = metrics["tpr"]
                current_fpr = metrics["fpr"]
                current_recall = metrics["recall"]

        logger.info("Melhor threshold: %.4f  Melhor F1: %.4f", best_thresh, best_f1)

        if best_f1 > best_f1_score:
            best_threshold = best_thresh
            best_precision = current_precision
            best_accuracy = current_accuracy
            best_tpr = current_tpr
            best_fpr = current_fpr
            best_recall = current_recall
            best_f1_score = best_f1
            best_hyperparameters = hyperparameters

    logger.info("Melhores hiperparâmetros Autoencoder: %s", best_hyperparameters)
    logger.info("Melhor F1-score (val): %.4f", best_f1_score)

    # ------------------------------------------------------------------
    # Treinamento final do Autoencoder com os melhores hiperparâmetros
    # ------------------------------------------------------------------
    ae_model = ae.Autoencoder(in_features, best_hyperparameters["DROPOUT_RATE"])
    ae_model.compile(
        learning_rate=best_hyperparameters["ALPHA"],
        weight_decay=best_hyperparameters["REGULARIZER"],
    )
    ae_model.fit(
        optimal_x_train_tensor,
        best_hyperparameters["NUM_EPOCHS"],
        best_hyperparameters["BATCH_SIZE"],
        X_val=benign_x_val_tensor,
        patience=best_hyperparameters["PATIENCE"],
        delta=best_hyperparameters["DELTA"],
    )

    # ------------------------------------------------------------------
    # Avaliação no conjunto de teste
    # ------------------------------------------------------------------
    test_anomaly_scores = ae.get_autoencoder_anomaly_scores(ae_model, optimal_x_test)
    metrics = ae.get_overall_metrics(optimal_y_test, test_anomaly_scores > best_threshold)

    end_time = time.time()
    data.get_time(start_time, end_time)

    logger.info("Features escolhidas (%d): %s", len(chosen_features), chosen_features)
    logger.info(
        "Métricas do ensemble (%s): %.4f",
        "GradientBoosting" if ensemble_type == "gb" else "RandomForest",
        pso.evaluate_fitness(ensemble_type, swarm[0], column_names, df, y, swarm[0].index, n_features=n_features),
    )
    logger.info(
        "Split: %.2f | n_estimators: %s | learning_rate: %s",
        swarm[0].position[0],
        swarm[0].position[-2],
        swarm[0].position[-1],
    )
    logger.info("Hiperparâmetros Autoencoder: %s | threshold: %.4f", best_hyperparameters, best_threshold)
    logger.info(
        "Métricas Autoencoder (teste): f1=%.4f precision=%.4f accuracy=%.4f recall=%.4f tpr=%.4f fpr=%.4f",
        metrics["f1-score"],
        metrics["precision"],
        metrics["accuracy"],
        metrics["recall"],
        metrics["tpr"],
        metrics["fpr"],
    )


if __name__ == "__main__":
    main()
