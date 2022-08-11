let baseUrl = '';//http root path

if(process.env.NODE_ENV == 'production'){//production
baseUrl = '/';

} else if (process.env.NODE_ENV == 'development') {//development
baseUrl = '/api'
} else if (process.env.NODE_ENV == 'test') {//test
baseUrl = '/api'

}

export {
baseUrl

}
