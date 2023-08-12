# DJL Serving plugin management

## Available Plugins

- [KServe plugin](../../plugins/kserve/README.md) - KServe V2 Protocol support
- [Management console plugin](../../plugins/management-console/README.md) - DJL Management console UI
- [Cache plugin](../../plugins/cache/README.md) - Provides additional options for caches
- [Static File plugin](../../plugins/static-file-plugin/README.md) - Allows DJL Serving to also serve static files
- [Plugin Management plugin](../../plugins/plugin-management-plugin/README.md) - Adds plugin management to the management API
 
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

