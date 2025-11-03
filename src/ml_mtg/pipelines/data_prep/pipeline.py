from kedro.pipeline import Pipeline, node, pipeline
from .nodes import normalize_frames, expand_decks, merge_cards, build_features, split_train_test

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=normalize_frames,
            inputs=["all_mtg_cards", "standard_decks"],
            outputs=["norm_cards", "norm_decks"],
            name="normalize_frames"
        ),
        node(
            func=expand_decks,
            inputs="norm_decks",
            outputs="deck_cards_intermediate",
            name="expand_decks"
        ),
        node(
            func=merge_cards,
            inputs=["deck_cards_intermediate", "norm_cards"],
            outputs="merged_cards_decks",
            name="merge_cards"
        ),
        node(
            func=build_features,
            inputs=["merged_cards_decks", "norm_decks"],
            outputs="mtg_deck_features",
            name="build_features"
        ),
        node(
            func=split_train_test,
            inputs=dict(
                features="mtg_deck_features",
                tier_positive_regex="params:tier_positive_regex",
                test_size="params:test_size",
                random_state="params:random_state"
            ),
            outputs=["X_train_cls","X_test_cls","y_train_cls","y_test_cls",
                    "X_train_reg","X_test_reg","y_train_reg","y_test_reg"],
            name="split_train_test"
        )
    ])
