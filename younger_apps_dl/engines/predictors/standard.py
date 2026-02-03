#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-03 16:36:10
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import time
import torch
import pathlib

from typing import Any, Literal, Callable
from pydantic import BaseModel, Field

from younger.commons.utils import no_operation

from younger_apps_dl.commons.utils import get_device_descriptor
from younger_apps_dl.commons.logging import logger
from younger_apps_dl.commons.checkpoint import load_checkpoint

from younger_apps_dl.engines import BaseEngine, register_engine


class RawOptions(BaseModel):
    load_dirpath: pathlib.Path = Field(..., description="Directory path to inputs on disk.")
    save_dirpath: pathlib.Path = Field(..., description="Directory path to outputs on disk.")


class APIOptions(BaseModel):
    host: str = Field("0.0.0.0", description="API server host address.")
    port: int = Field(8000, description="API server port number.")
    api_route: str = Field("/predict", description="API endpoint route for prediction.")
    docs_route: str = Field("/docs", description="API documentation route (FastAPI only).")


class CLIOptions(BaseModel):
    prompt: str = Field("Input: ", description="Prompt message for user input in interactive mode.")


class StandardPredictorOptions(BaseModel):
    # Checkpoint Options
    checkpoint_filepath: pathlib.Path  = Field(..., description='Path to load checkpoint.')
    source: Literal['raw', 'api', 'cli'] = Field('raw', description='Source of input data for prediction. '
                                                                    '\'raw\' indicates data is loaded from disk; '
                                                                    '\'api\' indicates data comes from a live API call; '
                                                                    '\'cli\' indicates data is passed from command-line interface.')

    # Device Options
    device_type: Literal['CPU', 'GPU'] = Field('GPU', description='Device type for model inference. Use CUDA_VISIBLE_DEVICES environment variable to control GPU selection.')

    raw: RawOptions | None = None
    api: APIOptions | None = None
    cli: CLIOptions | None = None


@register_engine('predictor', 'standard')
class StandardPredictor(BaseEngine[StandardPredictorOptions]):
    OPTIONS = StandardPredictorOptions

    def run(
        self,
        model: torch.nn.Module,
        predict_raw_fn: Callable[[pathlib.Path, pathlib.Path], None] | None = None,
        predict_api_fn: Callable[[dict], dict] | None = None,
        predict_cli_fn: Callable[[str], str] | None = None,
        initialize_fn: Callable[[torch.nn.Module], None] = no_operation,
    ) -> None:
        if self.options.device_type == 'GPU' and not torch.cuda.is_available():
            logger.warning('GPU requested but CUDA is not available. Falling back to CPU training.')
            self.options.device_type = 'CPU'
        if self.options.device_type == 'CPU':
            logger.info('Using CPU for evaluation.')

        checkpoint = load_checkpoint(self.options.checkpoint_filepath)

        device_descriptor = get_device_descriptor(self.options.device_type, 0)
        model.to(device=device_descriptor)

        logger.info(f'-> Using Device: {device_descriptor}')
        logger.info(f'-> Checkpoint from [Epoch/Step/Itr]@[{checkpoint.epoch}/{checkpoint.step}/{checkpoint.itr}].')

        logger.info(f'    v Loading Parameters ...')
        model.load_state_dict(checkpoint.model_state_dict)
        logger.info(f'    ^ Loaded.')

        if self.options.source == 'raw':
            self.predict_raw(
                model,
                predict_raw_fn,
                initialize_fn,
            )

        if self.options.source == 'api':
            self.predict_api(
                model,
                predict_api_fn,
                initialize_fn,
            )

        if self.options.source == 'cli':
            self.predict_cli(
                model,
                predict_cli_fn,
                initialize_fn
            )

    def predict_raw(
        self,
        model: torch.nn.Module,
        predict_raw_fn: Callable[[pathlib.Path, pathlib.Path], None],
        initialize_fn: Callable[[torch.nn.Module], None] = no_operation,
    ) -> None:
        logger.info(f'-> Load Raw From: {self.options.raw.load_dirpath}')

        initialize_fn(model)
        logger.info(f'-> Predicting ...')
        tic = time.time()
        model.eval()
        with torch.no_grad():
            predict_raw_fn(self.options.raw.load_dirpath, self.options.raw.save_dirpath)

        toc = time.time()

        logger.info(f'-> Finished. Overall Time Cost = {toc-tic:.2f}s)')

    def predict_api(
        self,
        model: torch.nn.Module,
        predict_api_fn: Callable[[dict], dict],
        initialize_fn: Callable[[torch.nn.Module], None] = no_operation,
    ) -> None:
        """
        TODO: This function currently supports only simple prediction APIs using FastAPI.
        For more complex scenarios, users may need to implement custom API servers.
        Future versions may include support for additional features such as authentication,
        rate limiting, and more advanced request/response handling.
        """
        logger.info(f'-> Starting FastAPI Server at {self.options.api.host}:{self.options.api.port} ...')
        logger.info(f'-> API Endpoint: {self.options.api.api_route}')
        initialize_fn(model)

        model.eval()

        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel as PydanticBaseModel
        import uvicorn

        app = FastAPI(title="Younger Apps DL - Launch Prediction API", version="0.1.0")

        class PredictRequest(PydanticBaseModel):
            input: dict
        
        class PredictResponse(PydanticBaseModel):
            output: dict

        @app.post(self.options.api.api_route, response_model=PredictResponse)
        async def predict_endpoint(request: PredictRequest):
            try:
                with torch.no_grad():
                    output = predict_api_fn(request.input)
                return PredictResponse(output=output)
            except Exception as exception:
                logger.error(f'-> Prediction Error: {exception}')
                raise HTTPException(status_code=500, detail=str(exception))

        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "model": "loaded"}

        logger.info(f'-> API Documentation Available at: http://{self.options.api.host}:{self.options.api.port}{self.options.api.docs_route}')
        logger.info(f'-> API Server will Remain Active. Press Ctrl+C to stop.')

        uvicorn.run(app, host=self.options.api.host, port=self.options.api.port, log_level="info")

        logger.info(f'-> API Server Stopped by User.')

    def predict_cli(
        self,
        model: torch.nn.Module,
        predict_cli_fn: Callable[[str], str],
        initialize_fn: Callable[[torch.nn.Module], None] = no_operation,
    ) -> None:
        logger.info(f'-> Starting Interactive CLI Prediction ...')
        logger.info(f'-> Type "exit" or "quit" to stop, or press Ctrl+C.')
        initialize_fn(model)

        model.eval()
        with torch.no_grad():
            while True:
                try:
                    cli_input = input(self.options.cli.prompt).strip()

                    if cli_input.lower() in ['exit', 'quit']:
                        logger.info(f'-> CLI Prediction Ended by User.')
                        break

                    if len(cli_input) == 0:
                        continue

                    tic = time.time()
                    cli_output = predict_cli_fn(cli_input)
                    toc = time.time()
                    
                    print(f"Output: {cli_output}")
                    logger.info(f'-> Prediction Completed in {toc-tic:.2f}s')

                except KeyboardInterrupt:
                    logger.info(f'-> CLI Prediction Interrupted by User.')
                    break
                except EOFError:
                    logger.info(f'-> CLI Prediction Ended (EOF).')
                    break
                except Exception as exception:
                    logger.error(f'-> Prediction Error: {exception}')
                    print(f"Error: {exception}")
