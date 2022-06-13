# DJL Serving plugin management

## Available Plugins

 - [Static File Plugin](../../plugins/static-file-plugin/README.md) - Allows DJL Serving to also serve static files
 - [Plugin Management Plugin](../../plugins/plugin-management-plugin/README.md) - Adds plugin management to the management API
 - [DJL Central](../../central/README.md) - DJL Central is added as a DJL Serving plugin

## Installing plug-ins

The model server looks for plug-ins during startup in the plugin folder and register this plug-ins.

The default plug-in folder is

```sh
{work-dir}/plugins
```

The plug-in folder can be configured with the 'plugin-folder' parameter in the server-config file.

example:
running model server with gradle using a specific config-file:

```sh
djl-serving -f ~/modelserver-config.properties
```

example config.properties file for djl-server

```sh
inference_address=http://127.0.0.1:8081
management_address=http://127.0.0.1:8081
plugin_folder=serving_plugins
```

